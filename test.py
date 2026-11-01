import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Hardware acceleration setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {device}")
if device == "cuda":
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True


def get_line_side(p1, p2, point):
    """Return which side of a directed line a point lies on."""
    return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])


def get_segment_intersection(p1, p2, q1, q2):
    """Return intersection fractions along two 2D segments, if they intersect."""
    movement_x = p2[0] - p1[0]
    movement_y = p2[1] - p1[1]
    line_x = q2[0] - q1[0]
    line_y = q2[1] - q1[1]
    denominator = movement_x * line_y - movement_y * line_x
    if abs(denominator) < 1e-9:
        return None

    offset_x = q1[0] - p1[0]
    offset_y = q1[1] - p1[1]
    movement_fraction = (
        offset_x * line_y - offset_y * line_x
    ) / denominator
    line_fraction = (
        offset_x * movement_y - offset_y * movement_x
    ) / denominator
    if 0 <= movement_fraction <= 1 and 0 <= line_fraction <= 1:
        return movement_fraction, line_fraction
    return None


def format_video_timestamp(frame_index, fps):
    """Format a frame position as elapsed video time."""
    total_milliseconds = round(frame_index / fps * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def normalized_feature(track):
    """Get a unit-length pretrained ReID embedding from a DeepSORT track."""
    feature = np.asarray(track.get_feature(), dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 1e-12 else None


def update_identity_gallery(identity, feature, max_prototypes=5):
    """Maintain quality-filtered appearance embeddings without prototype drift."""
    prototypes = identity.setdefault("prototypes", [])
    counts = identity.setdefault("prototype_counts", [])

    if not prototypes:
        prototypes.append(feature.copy())
        counts.append(1)
        return

    similarities = [float(np.dot(feature, p)) for p in prototypes]
    best_idx = int(np.argmax(similarities))
    best_sim = similarities[best_idx]

    # Ignore noisy or heavily occluded features
    if best_sim < 0.55:
        return

    # Update existing prototype using exponential moving average if highly similar
    if best_sim >= 0.85:
        updated = 0.90 * prototypes[best_idx] + 0.10 * feature
        norm = np.linalg.norm(updated)
        if norm > 1e-12:
            prototypes[best_idx] = updated / norm
        counts[best_idx] += 1
    # Store distinct pose/viewpoint if gallery isn't full
    elif len(prototypes) < max_prototypes:
        prototypes.append(feature.copy())
        counts.append(1)
    # Replace least-matched prototype only when new feature is sufficiently distinct
    else:
        replace_idx = int(np.argmin(counts))
        prototypes[replace_idx] = feature.copy()
        counts[replace_idx] = 1


def match_global_identity(feature, center, frame_index, fps, frame_shape,
                          identities, active_global_ids, assigned_global_ids):
    """Match a local track against historical ReID embeddings with realistic thresholds."""
    height, width = frame_shape[:2]
    frame_diagonal = math.hypot(width, height)
    candidates = []

    for global_id, identity in identities.items():
        if global_id in active_global_ids or global_id in assigned_global_ids:
            continue
        prototypes = identity.get("prototypes", [])
        if not prototypes:
            continue

        elapsed_frames = frame_index - identity.get("last_frame", frame_index)
        if elapsed_frames <= 0:
            continue

        distance = math.hypot(
            center[0] - identity["center"][0],
            center[1] - identity["center"][1],
        )
        # Scaled spatial search window allowing movement during long occlusions
        max_distance = min(frame_diagonal, 180 + (320 * elapsed_frames / fps))
        if distance > max_distance:
            continue

        similarity = max(float(np.dot(feature, p)) for p in prototypes)
        
        if similarity >= 0.72:
            candidates.append((similarity, global_id))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])

    # Ambiguity check: Only reject if top 2 candidates are virtually indistinguishable
    if len(candidates) > 1:
        top_sim, second_sim = candidates[0][0], candidates[1][0]
        if top_sim - second_sim < 0.08:
            return None

    return candidates[0][1]


def identity_similarity(feature, identity):
    prototypes = identity.get("prototypes", [])
    if not prototypes:
        return -1.0
    return max(float(np.dot(feature, prototype)) for prototype in prototypes)


def match_staff_profile(feature, staff_profiles):
    scores = [
        (
            max(float(np.dot(feature, prototype)) for prototype in profile["prototypes"]),
            staff_id,
        )
        for staff_id, profile in staff_profiles.items()
        if profile["prototypes"]
    ]
    scores.sort(reverse=True)
    if not scores or scores[0][0] < 0.70:
        return None
    if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.14:
        return None
    return scores[0][1]


def update_staff_profile(feature, staff_id, staff_profiles, frame_index,
                         last_update_frames):
    profile = staff_profiles[staff_id]
    best_index = int(
        np.argmax([float(np.dot(feature, prototype)) for prototype in profile["prototypes"]])
    )
    best_similarity = float(np.dot(feature, profile["prototypes"][best_index]))
    if best_similarity < 0.70 or frame_index - last_update_frames[staff_id] < 12:
        return

    last_update_frames[staff_id] = frame_index
    if best_similarity >= 0.90:
        updated = 0.95 * profile["prototypes"][best_index] + 0.05 * feature
        profile["prototypes"][best_index] = updated / np.linalg.norm(updated)
    elif len(profile["prototypes"]) < 8:
        profile["prototypes"].append(feature.copy())


def update_crossing_state(track_id, point, frame_index, crossing_states,
                          line_start, line_end, min_dist, fps,
                          confirmation_frames=3, cooldown_frames=39):
    state = crossing_states.get(track_id)
    current_side = get_line_side(line_start, line_end, point) / math.hypot(
        line_end[0] - line_start[0],
        line_end[1] - line_start[1],
    )
    if state is None:
        crossing_states[track_id] = {
            "zone": "inside" if current_side < 0 else "outside",
            "pending_zone": None,
            "pending_hits": 0,
            "last_point": point,
            "last_frame": frame_index,
        }
        return None

    previous_point = state["last_point"]
    previous_frame = state["last_frame"]
    state["last_point"] = point
    state["last_frame"] = frame_index

    previous_side = get_line_side(line_start, line_end, previous_point) / math.hypot(
        line_end[0] - line_start[0],
        line_end[1] - line_start[1],
    )

    side_changed = (
        previous_side < 0 <= current_side or current_side < 0 <= previous_side
    )
    intersection = get_segment_intersection(
        previous_point, point, line_start, line_end
    ) if side_changed else None
    if intersection is not None:
        movement_x = point[0] - previous_point[0]
        movement_y = point[1] - previous_point[1]
        elapsed_frames = max(1, frame_index - previous_frame)
        movement = math.hypot(
            point[0] - previous_point[0],
            point[1] - previous_point[1],
        )
        max_movement = max(60, 240 * elapsed_frames / fps)
        if movement <= max_movement:
            state["pending_zone"] = "inside" if current_side < 0 else "outside"
            state["pending_hits"] = 0
            movement_fraction, _ = intersection
            state["crossing_point"] = (
                previous_point[0] + movement_fraction * movement_x,
                previous_point[1] + movement_fraction * movement_y,
            )

    pending_zone = state["pending_zone"]
    if pending_zone is None:
        return None

    is_inside = current_side < 0
    if (pending_zone == "inside") != is_inside:
        state["pending_zone"] = None
        state["pending_hits"] = 0
        return None

    required_distance = min_dist
    required_hits = confirmation_frames
    if pending_zone == "outside":
        required_distance *= 0.65
        required_hits = 2

    if abs(current_side) < required_distance:
        state["pending_hits"] = 0
        return None

    state["pending_hits"] += 1
    if state["pending_hits"] < required_hits:
        return None

    last_crossing_frame = state.get("last_crossing_frame")
    crossing_direction = "in" if pending_zone == "inside" else "out"
    if (
        last_crossing_frame is not None
        and frame_index - last_crossing_frame < cooldown_frames
        and crossing_direction == state.get("last_crossing_direction")
    ):
        state["zone"] = pending_zone
        state["pending_zone"] = None
        state["pending_hits"] = 0
        return None

    state["zone"] = pending_zone
    state["pending_zone"] = None
    state["pending_hits"] = 0
    state["last_crossing_frame"] = frame_index
    state["last_crossing_direction"] = crossing_direction
    return crossing_direction


def register_counting_event(global_id, feature, direction, frame_index, fps,
                            event_profiles, global_event_profiles,
                            crossing_point=None,
                            similarity_threshold=0.80, dedupe_seconds=20):
    duplicate_window = round(fps * 0.75)
    reversal_window = round(fps * 3)
    if crossing_point is not None:
        for profile in event_profiles:
            previous_point = profile.get("last_point")
            if (
                profile["last_direction"] == direction
                and frame_index - profile["last_frame"] <= duplicate_window
                and previous_point is not None
                and math.hypot(
                    crossing_point[0] - previous_point[0],
                    crossing_point[1] - previous_point[1],
                ) <= 35
            ):
                print(
                    f"[COUNT-DEDUPE] frame={frame_index} direction={direction} "
                    f"near previous event at {previous_point}"
                )
                return None

    event_similarity_threshold = (
        0.90 if direction == "out" else similarity_threshold
    )
    profile_index = global_event_profiles.get(global_id)
    replaced_direction = None
    if profile_index is not None:
        profile = event_profiles[profile_index]
        profile_similarity = max(
            float(np.dot(feature, prototype))
            for prototype in profile["prototypes"]
        )
        if profile_similarity < event_similarity_threshold:
            profile_index = None
        elif (
            profile["last_direction"] != direction
            and frame_index - profile["last_frame"] <= reversal_window
            and crossing_point is not None
            and profile.get("last_point") is not None
            and math.hypot(
                crossing_point[0] - profile["last_point"][0],
                crossing_point[1] - profile["last_point"][1],
            ) <= 100
        ):
            replaced_direction = profile["last_direction"]

    if profile_index is None:
        candidates = []
        for index, profile in enumerate(event_profiles):
            elapsed = frame_index - profile["last_frame"]
            if (
                elapsed > round(fps * dedupe_seconds)
                or profile["last_direction"] == direction
            ):
                continue
            similarity = max(
                float(np.dot(feature, prototype))
                for prototype in profile["prototypes"]
            )
            if similarity >= event_similarity_threshold:
                candidates.append((similarity, index))

        candidates.sort(reverse=True)
        if candidates:
            if (
                len(candidates) == 1
                or candidates[0][0] - candidates[1][0] >= 0.06
            ):
                profile_index = candidates[0][1]

    if profile_index is None:
        profile_index = len(event_profiles)
        event_profiles.append(
            {
                "prototypes": [feature.copy()],
                "last_frame": frame_index,
                "last_direction": direction,
                "last_point": crossing_point,
            }
        )
    else:
        profile = event_profiles[profile_index]
        profile["last_frame"] = frame_index
        profile["last_direction"] = direction
        profile["last_point"] = crossing_point
        best_similarity = max(
            float(np.dot(feature, prototype))
            for prototype in profile["prototypes"]
        )
        if best_similarity >= 0.84:
            updated = 0.95 * profile["prototypes"][0] + 0.05 * feature
            profile["prototypes"][0] = updated / np.linalg.norm(updated)
        elif len(profile["prototypes"]) < 5:
            profile["prototypes"].append(feature.copy())

    global_event_profiles[global_id] = profile_index
    return direction, replaced_direction


def assign_global_id(raw_id, feature, center, frame_index, fps, frame_shape,
                     raw_to_global, identities, active_global_ids,
                     assigned_global_ids, raw_assignment_state, next_global_id):
    """Reassess local-to-global mappings and correct sustained ReID mismatches."""
    if raw_id in raw_to_global:
        global_id = raw_to_global[raw_id]
        identity = identities[global_id]
        own_similarity = identity_similarity(feature, identity)
        update_gallery = own_similarity >= 0.65
        challenger_id = match_global_identity(
            feature,
            center,
            frame_index,
            fps,
            frame_shape,
            identities,
            active_global_ids,
            assigned_global_ids,
        )
        state = raw_assignment_state.setdefault(raw_id, {})
        if challenger_id is not None:
            challenger_similarity = identity_similarity(
                feature, identities[challenger_id]
            )
            if (
                challenger_similarity >= 0.75
                and challenger_similarity - own_similarity >= 0.10
            ):
                if state.get("challenger_id") == challenger_id:
                    state["challenger_hits"] += 1
                else:
                    state["challenger_id"] = challenger_id
                    state["challenger_hits"] = 1

                if state["challenger_hits"] >= 3:
                    print(
                        f"[RE-ID] Corrected raw track {raw_id}: "
                        f"GID {global_id} -> GID {challenger_id}"
                    )
                    global_id = challenger_id
                    raw_to_global[raw_id] = global_id
                    state.clear()
                    update_gallery = True
            else:
                state.clear()
        else:
            state.clear()
    else:
        global_id = match_global_identity(
            feature,
            center,
            frame_index,
            fps,
            frame_shape,
            identities,
            active_global_ids,
            assigned_global_ids,
        )
        if global_id is None:
            global_id = next_global_id
            next_global_id += 1
            identities[global_id] = {"prototypes": [], "prototype_counts": []}
        raw_to_global[raw_id] = global_id
        update_gallery = True

    identity = identities[global_id]
    if update_gallery:
        update_identity_gallery(identity, feature)
    identity["center"] = center
    identity["last_frame"] = frame_index
    assigned_global_ids.add(global_id)
    return global_id, next_global_id


project_root = Path(__file__).resolve().parent
input_dir = project_root / "input"
models_dir = project_root / "models"
video_path = input_dir / "cctv_video.mp4"
output_dir = project_root / "outputs"
output_path = output_dir / "output_boundary_exclusion.mp4"
events_csv_path = output_path.with_suffix(".csv")
model_weights = models_dir / "yolov8x.pt"
reid_weights = models_dir / "osnet_ain_x1_0_msmt17.pth"

if not reid_weights.is_file():
    raise FileNotFoundError(f"OSNet ReID weights not found: {reid_weights}")
if not model_weights.is_file():
    raise FileNotFoundError(f"YOLO model weights not found: {model_weights}")

output_dir.mkdir(parents=True, exist_ok=True)

# Initialize YOLO Detection Model
model = YOLO(str(model_weights))
model.to(device)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Could not open input video: {video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if width <= 0 or height <= 0 or fps <= 0:
    cap.release()
    raise ValueError(f"Invalid video properties: {width}x{height} at {fps} FPS")

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)
if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create output video: {output_path}")

# Upgraded DeepSORT Parameters for Occlusion Robustness
tracker = DeepSort(
    max_age=max(90, round(fps * 6)),   # Store track memory longer (approx 6 seconds)
    n_init=2,                          # Confirm new tracks faster
    max_cosine_distance=0.40,          # Increased tolerance for pose/angle changes
    nn_budget=100,
    embedder="torchreid",
    embedder_model_name="osnet_ain_x1_0",
    embedder_wts=str(reid_weights),
    embedder_gpu=device == "cuda",
    bgr=True,
)
print("ReID model: OSNet-AIN trained on MSMT17")
print(f"Saving annotated video to: {output_path}")
print(f"Saving event CSV to: {events_csv_path}")

# Counting Line Configuration
line_start = (int(width * 0.821), int(height * 1.009))
line_end = (int(width * 0.904), int(height * 0.686))
min_dist = max(10, int(width * 0.03))

# Staff boundary shown for reference; crossing it does not assign staff identity.
staff_boundary_start = (int(width * 0.105), int(height * 0.18))
staff_boundary_end = (int(width * 0.185), int(height * 0.88))
staff_boundary_dx_per_y = (
    (staff_boundary_end[0] - staff_boundary_start[0])
    / (staff_boundary_end[1] - staff_boundary_start[1])
)
staff_recovery_top = int(height * 0.10)
staff_recovery_bottom = int(height * 0.94)
staff_recovery_left_at_top = round(
    staff_boundary_start[0]
    + staff_boundary_dx_per_y * (staff_recovery_top - staff_boundary_start[1])
)
staff_recovery_left_at_bottom = round(
    staff_boundary_end[0]
    + staff_boundary_dx_per_y * (staff_recovery_bottom - staff_boundary_end[1])
)
staff_seed_regions = [
    np.array(
        [
            (0, int(height * 0.50)),
            (int(width * 0.12), int(height * 0.50)),
            (int(width * 0.12), int(height * 0.78)),
            (0, int(height * 0.78)),
        ],
        dtype=np.int32,
    ),
    np.array(
        [
            (0, int(height * 0.32)),
            (int(width * 0.11), int(height * 0.32)),
            (int(width * 0.11), int(height * 0.49)),
            (0, int(height * 0.49)),
        ],
        dtype=np.int32,
    ),
]
staff_recovery_region = np.array(
    [
        (0, staff_recovery_top),
        (staff_recovery_left_at_top, staff_recovery_top),
        (staff_recovery_left_at_bottom, staff_recovery_bottom),
        (0, staff_recovery_bottom),
    ],
    dtype=np.int32,
)


# Global Tracker State Storage
raw_to_global = {}
identities = {}
raw_assignment_state = {}
staff_raw_ids = {}
staff_profiles = {}
staff_profile_last_update = {1: 0, 2: 0}
staff_reid_candidates = {}
staff_seed_candidates = [{}, {}]
next_global_id = 1
crossing_states = {}
counting_event_profiles = []
global_counting_event_profiles = {}
seed_pending = True
initial_raw_ids = set()
already_inside = 0
total_in = 0
total_out = 0
frame_index = 0

events_csv_file = None
try:
    events_csv_file = events_csv_path.open("w", newline="", encoding="utf-8")
    event_writer = csv.writer(events_csv_file)
    event_writer.writerow(("timestamp", "person_id", "event", "headcount"))

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        prediction = model.predict(
            frame,
            classes=[0],       # Person class only
            conf=0.15,         # Lower confidence to retain bounding boxes in occlusion
            imgsz=1280,
            device=device,
            half=device == "cuda",
            verbose=False,
        )[0]

        detections = []
        if prediction.boxes is not None:
            boxes = prediction.boxes.xyxy.cpu().numpy()
            confidences = prediction.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), confidence in zip(boxes, confidences):
                detections.append(
                    (
                        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        float(confidence),
                        "person",
                    )
                )

        tracks = tracker.update_tracks(detections, frame=frame)
        visible_tracks = [
            track for track in tracks
            if track.is_confirmed() and track.time_since_update == 0
        ]
        active_raw_ids = {str(track.track_id) for track in visible_tracks}
        active_global_ids = {
            raw_to_global[raw_id]
            for raw_id in active_raw_ids
            if raw_id in raw_to_global
        }
        assigned_global_ids = set()
        if seed_pending:
            initial_raw_ids.update(active_raw_ids)

        if len(staff_raw_ids) < 2 and frame_index < round(fps * 2):
            for track in visible_tracks:
                x1, y1, x2, y2 = map(int, track.to_ltrb(orig=True))
                footpoint = ((x1 + x2) // 2, y2)
                raw_id = str(track.track_id)
                for region_index, region in enumerate(staff_seed_regions):
                    if cv2.pointPolygonTest(region, footpoint, False) < 0:
                        continue
                    candidate = staff_seed_candidates[region_index].setdefault(
                        raw_id, {"hits": 0, "confidence": 0.0, "features": []}
                    )
                    candidate["hits"] += 1
                    candidate["confidence"] += float(track.get_det_conf() or 0.0)
                    feature = normalized_feature(track)
                    if feature is not None:
                        candidate["features"].append(feature)

            selected_staff = []
            for candidates in staff_seed_candidates:
                eligible = [
                    (data["confidence"] / data["hits"], raw_id, data)
                    for raw_id, data in candidates.items()
                    if data["hits"] >= 5
                ]
                if not eligible:
                    selected_staff = []
                    break
                selected_staff.append(max(eligible, key=lambda item: item[0]))

            if (
                len(selected_staff) == 2
                and selected_staff[0][1] != selected_staff[1][1]
            ):
                for staff_id, (_, raw_id, _) in enumerate(
                    selected_staff, start=1
                ):
                    staff_raw_ids[raw_id] = staff_id
                    candidate = staff_seed_candidates[staff_id - 1][raw_id]
                    mean_feature = np.mean(candidate["features"], axis=0)
                    norm = np.linalg.norm(mean_feature)
                    if norm > 1e-12:
                        staff_profiles[staff_id] = {
                            "prototypes": [(mean_feature / norm).astype(np.float32)]
                        }
                print(
                    "[STAFF] Locked only the two startup-position identities: "
                    + ", ".join(
                        f"STAFF {staff_id} (track {raw_id})"
                        for raw_id, staff_id in staff_raw_ids.items()
                    )
                )

        staff_seed_raw_ids = {
            raw_id
            for candidates in staff_seed_candidates
            for raw_id in candidates
        } if seed_pending and len(staff_raw_ids) < 2 else set()

        for track in visible_tracks:
            x1, y1, x2, y2 = map(int, track.to_ltrb(orig=True))
            x1, x2 = max(0, x1), min(width - 1, x2)
            y1, y2 = max(0, y1), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            footpoint = ((x1 + x2) // 2, y2)
            raw_id = str(track.track_id)
            feature = normalized_feature(track)
            in_staff_recovery_region = (
                cv2.pointPolygonTest(
                    staff_recovery_region,
                    (float(footpoint[0]), float(footpoint[1])),
                    False,
                )
                >= 0
            )

            staff_id = staff_raw_ids.get(raw_id)
            if (
                staff_id is None
                and feature is not None
                and len(staff_profiles) == 2
                and in_staff_recovery_region
            ):
                matched_staff_id = match_staff_profile(feature, staff_profiles)
                candidate = staff_reid_candidates.get(raw_id)
                if matched_staff_id is None:
                    staff_reid_candidates.pop(raw_id, None)
                else:
                    if candidate and candidate["staff_id"] == matched_staff_id:
                        candidate["hits"] += 1
                    else:
                        candidate = {"staff_id": matched_staff_id, "hits": 1}
                        staff_reid_candidates[raw_id] = candidate

                    active_staff_ids = {
                        staff_raw_ids[active_raw_id]
                        for active_raw_id in active_raw_ids
                        if active_raw_id in staff_raw_ids
                    }
                    if matched_staff_id in active_staff_ids:
                        continue
                    if candidate["hits"] >= 12:
                        staff_id = matched_staff_id
                        staff_raw_ids[raw_id] = staff_id
                        staff_reid_candidates.pop(raw_id, None)
                        print(
                            f"[STAFF] Recovered STAFF {staff_id} "
                            f"as track {raw_id} after ReID confirmation"
                        )
                    else:
                        cv2.putText(
                            frame,
                            "VERIFYING STAFF",
                            (x1, max(18, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 220, 255),
                            1,
                        )
                        continue

            if staff_id is not None:
                if feature is not None and staff_id in staff_profiles:
                    update_staff_profile(
                        feature,
                        staff_id,
                        staff_profiles,
                        frame_index,
                        staff_profile_last_update,
                    )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(
                    frame,
                    f"STAFF {staff_id}",
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 255),
                    2,
                )
                continue

            if in_staff_recovery_region:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 160, 180), 2)
                cv2.putText(
                    frame,
                    "STAFF SIDE",
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 220, 220),
                    1,
                )
                continue

            if seed_pending and raw_id in staff_seed_raw_ids:
                continue

            if feature is None:
                continue

            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            global_id, next_global_id = assign_global_id(
                raw_id,
                feature,
                center,
                frame_index,
                fps,
                frame.shape,
                raw_to_global,
                identities,
                active_global_ids,
                assigned_global_ids,
                raw_assignment_state,
                next_global_id,
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 110, 120), 2)
            cv2.putText(
                frame,
                f"ID: {global_id}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.circle(frame, (int(center[0]), y2), 4, (0, 0, 255), -1)

            crossing_direction = update_crossing_state(
                raw_id,
                (int(center[0]), y2),
                frame_index,
                crossing_states,
                line_start,
                line_end,
                min_dist,
                fps,
            )
            if seed_pending:
                continue
            if crossing_direction is not None:
                event_result = register_counting_event(
                    global_id,
                    feature,
                    crossing_direction,
                    frame_index,
                    fps,
                    counting_event_profiles,
                    global_counting_event_profiles,
                    crossing_states[raw_id].get("crossing_point"),
                )
                if event_result is not None:
                    accepted_direction, replaced_direction = event_result
                    if replaced_direction == "in":
                        total_in -= 1
                    elif replaced_direction == "out":
                        total_out -= 1
                    if replaced_direction is not None:
                        print(
                            f"[COUNT-CORRECT] frame={frame_index} "
                            f"replaced {replaced_direction.upper()} with "
                            f"{accepted_direction.upper()} for person {global_id}"
                        )
                    if accepted_direction == "in":
                        total_in += 1
                    else:
                        total_out += 1
                    print(
                        f"[{accepted_direction.upper()}] frame={frame_index} "
                        f"raw={raw_id} person={global_id} "
                        f"at={crossing_states[raw_id].get('crossing_point')} "
                        f"-> headcount {already_inside + total_in - total_out}"
                    )
                    event_writer.writerow(
                        (
                            format_video_timestamp(frame_index, fps),
                            global_id,
                            accepted_direction.upper(),
                            already_inside + total_in - total_out,
                        )
                    )
                    events_csv_file.flush()

        if seed_pending and len(staff_raw_ids) == 2:
            already_inside = len(initial_raw_ids - set(staff_raw_ids))
            seed_pending = False
            print(
                f"[BASELINE] {already_inside} customer(s) already inside "
                "after excluding the two locked staff"
            )

        headcount = already_inside + total_in - total_out
        cv2.line(
            frame,
            staff_boundary_start,
            staff_boundary_end,
            (255, 255, 0),
            3,
        )
        cv2.line(frame, line_start, line_end, (0, 0, 255), 4)

        # Draw HUD Panel
        panel_x, panel_y = 20, height - 180
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + 280, panel_y + 145), (35, 35, 35), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, f"IN: {total_in}", (panel_x + 20, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {total_out}", (panel_x + 20, panel_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"START INSIDE: {already_inside}", (panel_x + 20, panel_y + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        cv2.putText(frame, f"HEADCOUNT: {headcount}", (panel_x + 20, panel_y + 124),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)
        frame_index += 1
        if frame_index % 100 == 0:
            print(f"Processed {frame_index} frames...")
finally:
    if events_csv_file is not None:
        events_csv_file.close()
    cap.release()
    writer.release()

print(
    f"Complete: {frame_index} frames. Saved {output_path}. "
    f"Final IN: {total_in} | OUT: {total_out} | "
    f"Already inside: {already_inside} | Headcount: {headcount}"
)