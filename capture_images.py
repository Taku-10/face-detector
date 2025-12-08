"""
Video Image Capture Script

Captures high-quality images from videos based on different modes:
- "going": Face detection with smile detection and ranking
- "coming": Person detection filtering out guide in bottom left corner
"""

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np
import os
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

DEFAULT_GUIDE_REGION_WIDTH_RATIO = 0.4
DEFAULT_GUIDE_REGION_HEIGHT_RATIO = 0.8


def calculate_sharpness(frame: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    Higher values indicate sharper images.

    Args:
        frame: Input image frame (BGR format)

    Returns:
        Sharpness score (variance of Laplacian)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def is_blurry(frame: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Check if frame is blurry based on sharpness threshold.

    Args:
        frame: Input image frame
        threshold: Minimum sharpness value (default: 100.0)

    Returns:
        True if frame is blurry, False otherwise
    """
    sharpness = calculate_sharpness(frame)
    return sharpness < threshold


def detect_smile(
    face_landmarks, image_width: int, image_height: int
) -> Tuple[bool, float]:
    """
    Detect if a person is smiling using facial landmarks.
    Uses MediaPipe Face Mesh landmark positions to detect smile.

    Args:
        face_landmarks: MediaPipe face landmarks
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Tuple of (is_smiling: bool, smile_confidence: float)
    """
    if not face_landmarks or not face_landmarks.landmark:
        return False, 0.0

    try:
        landmarks = face_landmarks.landmark

        # MediaPipe Face Mesh has 468 landmarks
        # Mouth landmarks (approximate indices - may need adjustment):
        # Left mouth corner: ~61, Right mouth corner: ~291
        # Top lip center: ~13, Bottom lip center: ~14
        # Alternative: use outer mouth landmarks

        # Try to get mouth region landmarks
        # Use a more robust approach: find mouth region by looking for landmarks
        # in the lower face region

        # Get nose tip and chin for reference
        nose_tip_idx = 4  # Nose tip
        chin_idx = 152  # Chin

        if len(landmarks) <= max(nose_tip_idx, chin_idx):
            return False, 0.0

        nose_tip = landmarks[nose_tip_idx]
        chin = landmarks[chin_idx]

        # Find mouth landmarks in the region between nose and chin
        mouth_landmarks = []
        for i, landmark in enumerate(landmarks):
            # Mouth region is roughly between nose and chin
            if nose_tip.y < landmark.y < chin.y:
                # Check if it's in the horizontal center region (mouth area)
                if abs(landmark.x - nose_tip.x) < 0.15:  # Within 15% of face width
                    mouth_landmarks.append((i, landmark))

        if len(mouth_landmarks) < 4:
            # Fallback: use known approximate indices
            try:
                # Try common mouth landmark indices
                left_corner = landmarks[61] if len(landmarks) > 61 else None
                right_corner = landmarks[291] if len(landmarks) > 291 else None
                top_lip = landmarks[13] if len(landmarks) > 13 else None
                bottom_lip = landmarks[14] if len(landmarks) > 14 else None

                if all([left_corner, right_corner, top_lip, bottom_lip]):
                    mouth_landmarks = [
                        (61, left_corner),
                        (291, right_corner),
                        (13, top_lip),
                        (14, bottom_lip),
                    ]
                else:
                    return False, 0.0
            except (IndexError, AttributeError):
                return False, 0.0

        # Extract mouth corner and lip positions
        mouth_x_coords = [lm[1].x for lm in mouth_landmarks]
        mouth_y_coords = [lm[1].y for lm in mouth_landmarks]

        left_corner_x = min(mouth_x_coords)
        right_corner_x = max(mouth_x_coords)
        top_lip_y = min(mouth_y_coords)
        bottom_lip_y = max(mouth_y_coords)

        # Calculate mouth metrics
        mouth_width = (right_corner_x - left_corner_x) * image_width
        mouth_height = (bottom_lip_y - top_lip_y) * image_height
        mouth_center_y = (top_lip_y + bottom_lip_y) / 2 * image_height

        # Get corner positions
        left_corner_y = (
            next(
                (
                    lm[1].y
                    for lm in mouth_landmarks
                    if abs(lm[1].x - left_corner_x) < 0.01
                ),
                top_lip_y,
            )
            * image_height
        )
        right_corner_y = (
            next(
                (
                    lm[1].y
                    for lm in mouth_landmarks
                    if abs(lm[1].x - right_corner_x) < 0.01
                ),
                top_lip_y,
            )
            * image_height
        )

        # Smile detection: corners should be higher than center (upturned)
        corner_avg_y = (left_corner_y + right_corner_y) / 2
        corner_upturn = mouth_center_y - corner_avg_y

        # Mouth should be wider than tall (horizontal stretch)
        mouth_ratio = mouth_width / max(mouth_height, 1.0)

        # Thresholds
        min_upturn = 1.0  # pixels (adjusted for normalized coordinates)
        min_mouth_ratio = 2.0

        # Check if smiling
        is_smiling = corner_upturn > min_upturn and mouth_ratio > min_mouth_ratio

        # Calculate confidence
        upturn_score = min(1.0, max(0.0, corner_upturn / 5.0))
        ratio_score = min(1.0, max(0.0, (mouth_ratio - min_mouth_ratio) / 2.0))
        confidence = (upturn_score + ratio_score) / 2.0

        return is_smiling, confidence

    except (IndexError, AttributeError, ValueError):
        # If landmark access fails, return no smile
        return False, 0.0


def is_face_looking_at_camera(
    face_landmarks, image_width: int, image_height: int
) -> bool:
    """
    Check if face is looking at the camera (frontal view).
    Uses MediaPipe Face Mesh landmarks to determine face orientation.

    Args:
        face_landmarks: MediaPipe face landmarks
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        True if face appears to be looking at camera
    """
    if not face_landmarks or not face_landmarks.landmark:
        return False

    try:
        landmarks = face_landmarks.landmark

        if len(landmarks) < 10:
            return False

        # MediaPipe Face Mesh landmark indices (approximate):
        # Left eye outer corner: ~33, Right eye outer corner: ~263
        # Nose tip: ~4

        # Try to get eye and nose positions
        # Use a more robust method: find landmarks in eye and nose regions
        nose_tip = None
        left_eye = None
        right_eye = None

        # Nose tip is typically around index 4
        if len(landmarks) > 4:
            nose_tip = landmarks[4]

        # Find eye landmarks (they're typically in specific regions)
        # Left eye region: x < 0.5, y around 0.3-0.5
        # Right eye region: x > 0.5, y around 0.3-0.5
        eye_candidates = []
        for i, lm in enumerate(landmarks):
            if 0.2 < lm.y < 0.5:  # Eye region vertically
                if 0.2 < lm.x < 0.8:  # Eye region horizontally
                    eye_candidates.append((i, lm))

        # Separate left and right eyes
        if nose_tip:
            left_eyes = [c for c in eye_candidates if c[1].x < nose_tip.x]
            right_eyes = [c for c in eye_candidates if c[1].x > nose_tip.x]

            if left_eyes and right_eyes:
                # Use the outermost eye corners
                left_eye = min(left_eyes, key=lambda x: x[1].x)[1]
                right_eye = max(right_eyes, key=lambda x: x[1].x)[1]

        # Fallback: use known indices if available
        if not left_eye and len(landmarks) > 33:
            left_eye = landmarks[33]
        if not right_eye and len(landmarks) > 263:
            right_eye = landmarks[263]

        # Check all required landmarks are available
        if nose_tip is None or left_eye is None or right_eye is None:
            # If we can't find landmarks, assume frontal (conservative)
            return True

        # Calculate positions
        left_eye_x = left_eye.x * image_width
        right_eye_x = right_eye.x * image_width
        nose_x = nose_tip.x * image_width

        # Face center (between eyes)
        face_center_x = (left_eye_x + right_eye_x) / 2

        # Check if nose is centered (frontal view indicator)
        eye_distance = abs(right_eye_x - left_eye_x)
        nose_offset = abs(nose_x - face_center_x)

        # Nose should be close to center (within 25% of eye distance)
        # Also check that eyes are roughly symmetric
        is_centered = nose_offset < (eye_distance * 0.25)
        is_symmetric = eye_distance > 0  # Basic check

        return is_centered and is_symmetric

    except (IndexError, AttributeError, ValueError):
        # Conservative: if we can't determine, assume frontal
        return True


def are_eyes_open(
    face_landmarks, image_width: int, image_height: int
) -> Tuple[bool, float]:
    """
    Heuristic check for whether both eyes appear open.

    Uses a simple eye-aspect-ratio style metric on a subset of Face Mesh
    landmarks. Returns (eyes_open, confidence).

    This is intentionally conservative – if landmarks are missing or the
    geometry looks weird, we default to eyes_open=True with low confidence
    so that we don't accidentally discard otherwise good frames.
    """
    if not face_landmarks or not face_landmarks.landmark:
        return True, 0.0

    try:
        lm = face_landmarks.landmark

        # Common Face Mesh indices (approximate):
        # Right eye: 33 (outer), 159 (upper), 145 (lower)
        # Left eye: 263 (outer), 386 (upper), 374 (lower)
        needed_indices = [33, 159, 145, 263, 386, 374]
        if len(lm) <= max(needed_indices):
            return True, 0.0

        def eye_open_score(outer_idx: int, upper_idx: int, lower_idx: int) -> float:
            outer = lm[outer_idx]
            upper = lm[upper_idx]
            lower = lm[lower_idx]
            eye_width = abs(outer.x - ((upper.x + lower.x) / 2.0)) * image_width
            eye_height = abs(upper.y - lower.y) * image_height
            if eye_width <= 0:
                return 0.0
            ratio = eye_height / eye_width
            # Typical "open" is a moderate ratio; extremely small -> closed/blink.
            # Map ratio in [0.02, 0.08+] to [0, 1].
            min_r, max_r = 0.02, 0.08
            if ratio <= min_r:
                return 0.0
            if ratio >= max_r:
                return 1.0
            return (ratio - min_r) / (max_r - min_r)

        right_score = eye_open_score(33, 159, 145)
        left_score = eye_open_score(263, 386, 374)
        score = min(right_score, left_score)

        # Consider eyes open if both eyes have reasonable openness
        eyes_open = score > 0.3
        return eyes_open, float(score)

    except (IndexError, AttributeError, ValueError):
        # If we can't compute reliably, don't penalize the frame
        return True, 0.0


def is_in_guide_region(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    width_ratio: float = DEFAULT_GUIDE_REGION_WIDTH_RATIO,
    height_ratio: float = DEFAULT_GUIDE_REGION_HEIGHT_RATIO,
) -> bool:
    """
    Check if bounding box is in the guide region (bottom left corner).

    Args:
        bbox: Bounding box (x, y, w, h)
        frame_width: Frame width
        frame_height: Frame height
        width_ratio: Fraction of frame width covered by guide region
        height_ratio: Fraction of frame height covered by guide region

    Returns:
        True if bbox is in guide region
    """
    x, y, w, h = bbox
    bbox_right = x + w
    bbox_bottom = y + h

    width_ratio = min(max(width_ratio, 0.05), 1.0)
    height_ratio = min(max(height_ratio, 0.05), 1.0)

    # Guide region: configurable fraction on bottom-left of frame
    guide_region_width = frame_width * width_ratio
    guide_region_height = frame_height * height_ratio
    guide_region_y_start = frame_height - guide_region_height

    # Treat as guide only when the entire box stays inside the guide region
    return bbox_right <= guide_region_width and bbox_bottom >= guide_region_y_start


def capture_images_from_video(
    video_path: str,
    mode: Optional[str] = None,
    min_pictures: Optional[int] = None,
    max_pictures: Optional[int] = None,
    platform_number: Optional[int] = None,
    output_dir: Optional[str] = None,
    sharpness_threshold: float = 100.0,
    show_progress: bool = False,
    show_frames: bool = False,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Capture images from video based on specified mode and criteria.

    Args:
        video_path: Path to input video file
        mode: Detection mode ("going", "coming", or "group")
            - "going": Face detection with smile detection and ranking
            - "coming": Person detection filtering out guide in bottom left corner
            - "group": Detects multiple faces and captures frames with most faces visible
            - If None, defaults to "going"
        min_pictures: Minimum number of pictures to capture (defaults to 5 if None)
        max_pictures: Maximum number of pictures to capture (defaults to 10 if None)
        platform_number: Optional identifier carried through to the result (no longer alters behavior)
        output_dir: Output directory for images
            - If None, automatically generates: {video_name}-images in same directory as video
            - Example: "vid1.mp4" -> "vid1-images/"
        sharpness_threshold: Minimum sharpness value to accept frame
        show_progress: If True, display progress information
        show_frames: If True, displays frames with detection overlay in real-time
            - For "coming" mode, guide-region overlay uses configured width/height ratios
        start_time: Optional start time in seconds to limit processing to a specific segment
            - If provided, only processes frames from start_time onwards
        end_time: Optional end time in seconds to limit processing to a specific segment
            - If provided, only processes frames up to end_time
            - If both start_time and end_time are provided, only processes that time range

    Returns:
        Dictionary with capture results (Any type for flexibility)
    """
    guide_region_width_ratio = DEFAULT_GUIDE_REGION_WIDTH_RATIO
    guide_region_height_ratio = DEFAULT_GUIDE_REGION_HEIGHT_RATIO

    # Apply defaults if unspecified
    if mode is None:
        mode = "going"
    if min_pictures is None:
        min_pictures = 5
    if max_pictures is None:
        max_pictures = 10

    # Validate mode
    if mode not in ["going", "coming", "group"]:
        return {
            "success": False,
            "error": f"Invalid mode: {mode}. Must be 'going', 'coming', or 'group'",
        }

    # Validate video file
    if not os.path.exists(video_path):
        return {"success": False, "error": f"Video file not found: {video_path}"}

    # Setup output directory
    if output_dir is None:
        video_name = Path(video_path).stem
        video_dir = os.path.dirname(os.path.abspath(video_path))
        output_dir = os.path.join(video_dir, f"{video_name}-images")

    os.makedirs(output_dir, exist_ok=True)

    # Clean up old auto-generated images for this video so the directory
    # only reflects the current run. We only delete files that match our
    # naming patterns, to avoid touching anything custom the user put there.
    for name in os.listdir(output_dir):
        lower = name.lower()
        if (
            lower.startswith("frame_")
            and lower.endswith(".jpg")
        ) or (
            lower.startswith("extra_after_start_")
            and lower.endswith(".jpg")
        ) or (
            lower.startswith("extra_before_end_")
            and lower.endswith(".jpg")
        ):
            try:
                os.remove(os.path.join(output_dir, name))
            except OSError:
                # If a file cannot be removed, skip it and continue
                pass

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "Could not open video file"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0.0
    guide_region_width_px = int(frame_width * guide_region_width_ratio)
    guide_region_height_px = int(frame_height * guide_region_height_ratio)
    guide_region_y_start_px = frame_height - guide_region_height_px
    guide_region_margin_px = max(10, int(frame_width * 0.02))

    # Initialize MediaPipe
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,  # Full-range model
        min_detection_confidence=0.5,
    )

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize background subtractor for person detection
    # (used in coming/group modes, and as a low-priority fallback for going)
    bg_subtractor = None
    if mode in ["coming", "group", "going"]:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

    # Sample every N frames (process ~10 frames per second for efficiency)
    sample_interval = max(1, int(fps * 0.1))

    # Store candidate frames with scores
    candidate_frames: List[Dict] = []

    frame_count = 0
    images_captured = 0
    last_detection_info = None  # Ensure defined before processing loop

    # Calculate frame range if start_time/end_time are provided
    start_frame = None
    end_frame = None
    if start_time is not None:
        start_frame = int(start_time * fps)
    if end_time is not None:
        end_frame = int(end_time * fps)

    # Seek to start frame if specified
    if start_frame is not None and start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame

    if show_progress:
        print(f"Processing video: {video_path}")
        print(f"Mode: {mode}")
        print(f"Target: {min_pictures}-{max_pictures} images")
        if start_time is not None or end_time is not None:
            time_range = f"Time range: {start_time or 0:.2f}s - {end_time or video_duration:.2f}s"
            print(time_range)
        print(f"Processing {total_frames} frames...")

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = frame_count / fps

        # Skip if before start_time
        if start_time is not None and frame_time < start_time:
            frame_count += 1
            continue

        # Stop if past end_time
        if end_time is not None and frame_time > end_time:
            break

        # Sample frames for efficiency
        if frame_count % sample_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Check blurriness first (reject blurry frames early)
            if is_blurry(frame, sharpness_threshold):
                frame_count += 1
                continue

            sharpness_score = calculate_sharpness(frame)
            last_detection_info = None  # Initialize for visualization

            if mode == "going":
                # GOING MODE: Face detection with priorities and person fallback
                face_results = face_detection.process(rgb_frame)

                has_face_candidate = False

                if face_results.detections:
                    # Find largest face
                    primary_detection = max(
                        face_results.detections,
                        key=lambda det: (
                            det.location_data.relative_bounding_box.width
                            * det.location_data.relative_bounding_box.height
                        ),
                    )

                    bbox = primary_detection.location_data.relative_bounding_box
                    confidence = primary_detection.score[0]

                    # Convert to pixel coordinates
                    x = int(bbox.xmin * frame_width)
                    y = int(bbox.ymin * frame_height)
                    w = int(bbox.width * frame_width)
                    h = int(bbox.height * frame_height)

                    # Ensure within bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame_width - x)
                    h = min(h, frame_height - y)

                    # Check if face is large enough (at least 6% of frame width)
                    min_face_width = int(frame_width * 0.06)
                    if w >= min_face_width and h >= min_face_width:
                        # Get face mesh for smile detection and frontal check
                        face_mesh_results = face_mesh.process(rgb_frame)

                        is_frontal = False
                        is_smiling = False
                        smile_confidence = 0.0
                        eyes_open = True
                        eyes_open_conf = 0.0

                        if face_mesh_results.multi_face_landmarks:
                            face_landmarks = face_mesh_results.multi_face_landmarks[0]
                            is_frontal = is_face_looking_at_camera(
                                face_landmarks, frame_width, frame_height
                            )
                            is_smiling, smile_confidence = detect_smile(
                                face_landmarks, frame_width, frame_height
                            )
                            eyes_open, eyes_open_conf = are_eyes_open(
                                face_landmarks, frame_width, frame_height
                            )

                        # Priority tiers for GOING mode (higher is better):
                        #   4: Face + frontal + smiling + eyes open
                        #   3: Face + frontal (+/- smile/eyes)
                        #   2: Face (non‑frontal) but still valid size
                        #   1: Person only (no face) – handled via bg_subtractor below
                        if is_frontal and is_smiling and eyes_open:
                            priority_level = 4
                        elif is_frontal:
                            priority_level = 3
                        else:
                            priority_level = 2

                        # Calculate score: base score from confidence and sharpness
                        # plus bonuses for smiling and eyes open
                        base_score = (
                            confidence * 0.5 + (sharpness_score / 500.0) * 0.25
                        )
                        smile_bonus = smile_confidence * 0.15 if is_smiling else 0.0
                        eyes_bonus = eyes_open_conf * 0.1 if eyes_open else 0.0
                        total_score = base_score + smile_bonus + eyes_bonus

                        candidate_frames.append(
                            {
                                "frame": frame.copy(),
                                "frame_count": frame_count,
                                "time": frame_time,
                                "score": total_score,
                                "priority_level": priority_level,
                                "confidence": confidence,
                                "sharpness": sharpness_score,
                                "is_smiling": is_smiling,
                                "smile_confidence": smile_confidence,
                                "eyes_open": eyes_open,
                                "eyes_open_confidence": eyes_open_conf,
                                "bbox": (x, y, w, h),
                            }
                        )
                        has_face_candidate = True

                        # Store detection info for visualization
                        last_detection_info = {
                            "bbox": (x, y, w, h),
                            "confidence": confidence,
                            "is_smiling": is_smiling,
                            "smile_confidence": smile_confidence,
                            "eyes_open": eyes_open,
                            "eyes_open_confidence": eyes_open_conf,
                            "sharpness": sharpness_score,
                            "score": total_score,
                            "is_frontal": is_frontal,
                            "priority_level": priority_level,
                        }

                # If we didn't get a face candidate this frame, fall back to
                # low‑priority person detection so that we still capture
                # something for this time bucket in very hard cases.
                if not has_face_candidate and bg_subtractor is not None:
                    fg_mask = bg_subtractor.apply(frame)

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                    contours, _ = cv2.findContours(
                        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)
                        min_area = (frame_width * frame_height) * 0.01

                        if area >= min_area:
                            x, y, w, h = cv2.boundingRect(largest_contour)

                            # Low‑priority score based mostly on area + sharpness
                            area_score = (area / (frame_width * frame_height)) * 0.4
                            sharp_score = (sharpness_score / 500.0) * 0.3
                            total_score = area_score + sharp_score

                            candidate_frames.append(
                                {
                                    "frame": frame.copy(),
                                    "frame_count": frame_count,
                                    "time": frame_time,
                                    "score": total_score,
                                    "priority_level": 1,
                                    "sharpness": sharpness_score,
                                    "has_face": False,
                                    "bbox": (x, y, w, h),
                                }
                            )

                            last_detection_info = {
                                "bbox": (x, y, w, h),
                                "has_face": False,
                                "sharpness": sharpness_score,
                                "score": total_score,
                                "is_frontal": False,
                                "priority_level": 1,
                            }

            elif mode == "coming":
                # COMING MODE: Person detection filtering out guide
                # Use background subtraction for person detection
                if bg_subtractor is None:
                    frame_count += 1
                    continue
                # Type assertion: bg_subtractor is not None after check
                assert bg_subtractor is not None
                fg_mask = bg_subtractor.apply(frame)

                # Morphological operations to reduce noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                # Find contours
                contours, _ = cv2.findContours(
                    fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    # Find largest contour (person)
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)

                    # Threshold for significant person (at least 2% of frame)
                    min_area = (frame_width * frame_height) * 0.02

                    if area >= min_area:
                        x, y, w, h = cv2.boundingRect(largest_contour)

                        # Check if this is NOT in the guide region (bottom left)
                        if not is_in_guide_region(
                            (x, y, w, h),
                            frame_width,
                            frame_height,
                            guide_region_width_ratio,
                            guide_region_height_ratio,
                        ):
                            # Also check if there's a face detected (to ensure rider, not just guide)
                            face_results = face_detection.process(rgb_frame)
                            has_face = False

                            if face_results.detections:
                                # Check if any detected face is NOT in guide region
                                for det in face_results.detections:
                                    bbox = det.location_data.relative_bounding_box
                                    face_x = int(bbox.xmin * frame_width)
                                    face_y = int(bbox.ymin * frame_height)
                                    face_w = int(bbox.width * frame_width)
                                    face_h = int(bbox.height * frame_height)

                                    if not is_in_guide_region(
                                        (face_x, face_y, face_w, face_h),
                                        frame_width,
                                        frame_height,
                                        guide_region_width_ratio,
                                        guide_region_height_ratio,
                                    ):
                                        has_face = True
                                        break

                            # Accept frame if person detected outside guide region
                            # Prefer frames with faces (rider) but also accept person detections
                            score = (area / (frame_width * frame_height)) * 0.5
                            if has_face:
                                score += 0.5

                            # Add sharpness component
                            score += (sharpness_score / 500.0) * 0.2

                            person_outside_guide = (x + w) > (
                                guide_region_width_px + guide_region_margin_px
                            )
                            if not has_face and not person_outside_guide:
                                last_detection_info = {
                                    "bbox": (x, y, w, h),
                                    "has_face": has_face,
                                    "sharpness": sharpness_score,
                                    "score": score,
                                    "area": area,
                                    "rejected": "guide_only",
                                }
                                continue

                            candidate_frames.append(
                                {
                                    "frame": frame.copy(),
                                    "frame_count": frame_count,
                                    "time": frame_time,
                                    "score": score,
                                    # For coming mode, we treat all accepted
                                    # detections as the same tier (person/face),
                                    # but still set a priority_level in case the
                                    # selection logic wants to use it later.
                                    "priority_level": 2 if has_face else 1,
                                    "sharpness": sharpness_score,
                                    "has_face": has_face,
                                    "bbox": (x, y, w, h),
                                }
                            )

                            # Store detection info for visualization
                            last_detection_info = {
                                "bbox": (x, y, w, h),
                                "has_face": has_face,
                                "sharpness": sharpness_score,
                                "score": score,
                                "area": area,
                                "rejected": None,
                            }

            elif mode == "group":
                # GROUP MODE: Detect both people (person detection) and faces
                # Count total people/faces to capture frames with most people visible

                # 1. Detect people using background subtraction
                person_count = 0
                person_bboxes = []

                if bg_subtractor is not None:
                    fg_mask = bg_subtractor.apply(frame)

                    # Morphological operations to reduce noise
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                    # Find contours
                    contours, _ = cv2.findContours(
                        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        # Filter contours by area to find people
                        min_area = (frame_width * frame_height) * 0.02  # 2% of frame
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area >= min_area:
                                x, y, w, h = cv2.boundingRect(contour)
                                person_bboxes.append((x, y, w, h))
                                person_count += 1

                # 2. Detect faces
                face_results = face_detection.process(rgb_frame)
                face_count = 0
                face_bboxes = []
                valid_faces = []

                if face_results.detections:
                    for det in face_results.detections:
                        bbox = det.location_data.relative_bounding_box
                        confidence = det.score[0]

                        # Convert to pixel coordinates
                        x = int(bbox.xmin * frame_width)
                        y = int(bbox.ymin * frame_height)
                        w = int(bbox.width * frame_width)
                        h = int(bbox.height * frame_height)

                        # Ensure within bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame_width - x)
                        h = min(h, frame_height - y)

                        # Check if face is large enough (at least 3% of frame width for groups)
                        min_face_width = int(frame_width * 0.03)
                        if (
                            w >= min_face_width
                            and h >= min_face_width
                            and confidence >= 0.5
                        ):
                            valid_faces.append(
                                {
                                    "bbox": (x, y, w, h),
                                    "confidence": confidence,
                                }
                            )
                            face_bboxes.append((x, y, w, h))
                            face_count += 1

                # 3. Count total people/faces (use maximum of person_count and face_count, or sum)
                # We'll use the maximum to avoid double-counting, but prefer faces when available
                total_count_int = max(person_count, face_count)
                total_count = float(total_count_int)

                # If we have both person and face detections, we can be more confident
                # So we add a small bonus if both detection methods agree
                if person_count > 0 and face_count > 0:
                    # Use the higher count, but add a small bonus for agreement
                    total_count = float(max(person_count, face_count)) + 0.5

                # Only consider frames with at least 2 people/faces (group requirement)
                if total_count >= 2:
                    # Calculate score: primarily based on total count
                    # More people/faces = higher score
                    count_score = total_count * 0.4  # 0.4 per person/face

                    # Average confidence of faces (if any)
                    avg_confidence = 0.0
                    if len(valid_faces) > 0:
                        avg_confidence = sum(
                            f["confidence"] for f in valid_faces
                        ) / len(valid_faces)
                    confidence_score = avg_confidence * 0.2

                    # Sharpness component
                    sharpness_score_component = (sharpness_score / 500.0) * 0.2

                    # Bonus for having both person and face detections
                    detection_method_bonus = (
                        0.2 if (person_count > 0 and face_count > 0) else 0.0
                    )

                    total_score = (
                        count_score
                        + confidence_score
                        + sharpness_score_component
                        + detection_method_bonus
                    )

                    # Priority tiers for GROUP mode:
                    #   3: Faces AND people detected
                    #   2: Faces only
                    #   1: People only
                    if person_count > 0 and face_count > 0:
                        priority_level = 3
                    elif face_count > 0:
                        priority_level = 2
                    else:
                        priority_level = 1

                    candidate_frames.append(
                        {
                            "frame": frame.copy(),
                            "frame_count": frame_count,
                            "time": frame_time,
                            "score": total_score,
                            "priority_level": priority_level,
                            "person_count": person_count,
                            "face_count": face_count,
                            "total_count": total_count,
                            "avg_confidence": avg_confidence,
                            "sharpness": sharpness_score,
                            "person_bboxes": person_bboxes,
                            "face_bboxes": face_bboxes,
                        }
                    )

                    # Store detection info for visualization
                    last_detection_info = {
                        "person_count": person_count,
                        "face_count": face_count,
                        "total_count": total_count,
                        "person_bboxes": person_bboxes,
                        "face_bboxes": face_bboxes,
                        "avg_confidence": avg_confidence,
                        "sharpness": sharpness_score,
                        "score": total_score,
                    }
                else:
                    last_detection_info = None

        # Create display frame with overlays if needed
        if show_frames:
            display_frame = frame.copy()

            if mode == "going":
                # Draw face/person detection with priority tiers
                if last_detection_info and last_detection_info.get("bbox") is not None:
                    x, y, w, h = last_detection_info["bbox"]
                    sharpness = last_detection_info.get("sharpness", 0.0)
                    score = last_detection_info.get("score", 0.0)
                    priority = last_detection_info.get("priority_level", 1)

                    if last_detection_info.get("has_face") is False:
                        # Person‑only fallback visualization
                        color = (0, 165, 255)
                        label = "Person (fallback)"
                    else:
                        confidence = last_detection_info.get("confidence", 0.0)
                        is_smiling = last_detection_info.get("is_smiling", False)
                        smile_conf = last_detection_info.get("smile_confidence", 0.0)
                        eyes_open = last_detection_info.get("eyes_open", True)
                        eyes_conf = last_detection_info.get(
                            "eyes_open_confidence", 0.0
                        )

                        # Draw face bounding box
                        color = (0, 255, 0) if is_smiling else (255, 0, 0)
                        label = f"Face ({confidence:.2f})"
                        if is_smiling:
                            label += f" | Smile ({smile_conf:.2f})"
                        if eyes_open:
                            label += f" | Eyes ({eyes_conf:.2f})"

                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
                    cv2.putText(
                        display_frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    info_text = [
                        f"Time: {frame_time:.2f}s | Frame: {frame_count}",
                        f"Mode: {mode} | Candidates: {len(candidate_frames)}",
                        f"Sharpness: {sharpness:.1f} | Score: {score:.3f}",
                        f"Tier: {priority}",
                    ]
                    for i, text in enumerate(info_text):
                        cv2.putText(
                            display_frame,
                            text,
                            (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                else:
                    # No detection
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_time:.2f}s | Searching...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            elif mode == "coming":
                # Draw person detection and guide region
                # Draw guide region (bottom left)
                cv2.rectangle(
                    display_frame,
                    (0, guide_region_y_start_px),
                    (guide_region_width_px, frame_height),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "Guide Region",
                    (10, guide_region_y_start_px + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

                if last_detection_info:
                    x, y, w, h = last_detection_info["bbox"]
                    has_face = last_detection_info["has_face"]
                    sharpness = last_detection_info["sharpness"]
                    score = last_detection_info["score"]
                    rejected_reason = last_detection_info.get("rejected")

                    # Draw person bounding box
                    color = (0, 255, 0) if rejected_reason is None else (0, 0, 255)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)

                    # Draw label
                    if rejected_reason == "guide_only":
                        label = "Guide (filtered)"
                    else:
                        label = f"Person ({'Rider' if has_face else 'No Face'})"
                    cv2.putText(
                        display_frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    # Add info overlay
                    info_text = [
                        f"Time: {frame_time:.2f}s | Frame: {frame_count}",
                        f"Mode: {mode} | Candidates: {len(candidate_frames)}",
                        f"Sharpness: {sharpness:.1f} | Score: {score:.3f}",
                        (
                            "Status: ACCEPTED"
                            if rejected_reason is None
                            else "Status: FILTERED"
                        ),
                    ]
                    for i, text in enumerate(info_text):
                        cv2.putText(
                            display_frame,
                            text,
                            (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                else:
                    # No detection
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_time:.2f}s | Searching...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            elif mode == "group":
                # Draw all detected people and faces
                if last_detection_info:
                    person_count = last_detection_info["person_count"]
                    face_count = last_detection_info["face_count"]
                    total_count = last_detection_info["total_count"]
                    person_bboxes = last_detection_info["person_bboxes"]
                    face_bboxes = last_detection_info["face_bboxes"]
                    avg_confidence = last_detection_info["avg_confidence"]
                    sharpness = last_detection_info["sharpness"]
                    score = last_detection_info["score"]

                    # Draw bounding boxes for all people (orange)
                    for i, (x, y, w, h) in enumerate(person_bboxes):
                        color = (0, 165, 255)  # Orange for people
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(
                            display_frame,
                            f"Person {i + 1}",
                            (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                    # Draw bounding boxes for all faces (green)
                    for i, (x, y, w, h) in enumerate(face_bboxes):
                        color = (0, 255, 0)  # Green for faces
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(
                            display_frame,
                            f"Face {i + 1}",
                            (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                    # Add info overlay
                    info_text = [
                        f"Time: {frame_time:.2f}s | Frame: {frame_count}",
                        f"Mode: {mode} | Candidates: {len(candidate_frames)}",
                        f"People: {person_count} | Faces: {face_count} | Total: {total_count:.1f}",
                        f"Avg Confidence: {avg_confidence:.2f} | Sharpness: {sharpness:.1f}",
                        f"Score: {score:.3f} | Status: ACCEPTED",
                    ]
                    for i, text in enumerate(info_text):
                        cv2.putText(
                            display_frame,
                            text,
                            (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                else:
                    # No detection or less than 2 people/faces
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_time:.2f}s | Searching for groups (min 2 people/faces)...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            # Show frame
            cv2.imshow("Image Capture - Detection", display_frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                cap.release()
                face_detection.close()
                face_mesh.close()
                cv2.destroyAllWindows()
                return {
                    "success": False,
                    "error": "Processing stopped by user",
                }

        frame_count += 1

        if show_progress and frame_count % (fps * 5) == 0:  # Every 5 seconds
            print(
                f"Processed {frame_count}/{total_frames} frames, "
                f"found {len(candidate_frames)} candidates"
            )

    cap.release()
    face_detection.close()
    face_mesh.close()
    if show_frames:
        cv2.destroyAllWindows()

    # Sort candidates by priority then score (highest first).
    # If priority_level is missing (older candidates), treat as 1.
    candidate_frames.sort(
        key=lambda x: (x.get("priority_level", 1), x["score"]), reverse=True
    )

    # Enforce at most one photo per whole-second bucket:
    # e.g. only one frame between t=1.00–1.99s, one between 2.00–2.99s, etc.
    # We iterate in (priority, score) order so each second keeps its "best"
    # quality frame first, then falls back to weaker ones if needed.
    per_second_selection: List[Dict[str, Any]] = []
    used_seconds = set()
    for cand in candidate_frames:
        second_bucket = int(cand["time"])
        if second_bucket in used_seconds:
            continue
        per_second_selection.append(cand)
        used_seconds.add(second_bucket)
        if len(per_second_selection) >= max_pictures:
            break

    if len(per_second_selection) < min_pictures:
        return {
            "success": False,
            "error": f"Only found {len(candidate_frames)} valid frames, "
            f"but {min_pictures} required",
            "candidates_found": len(candidate_frames),
            "output_dir": output_dir,
        }

    # Final list to save
    selected_frames = per_second_selection[:max_pictures]

    # Capture selected frames
    captured_files = []
    for candidate in selected_frames:
        frame = candidate["frame"]
        frame_time = candidate["time"]

        # Generate filename
        filename = f"frame_{candidate['frame_count']:06d}_t{frame_time:.2f}s.jpg"
        filepath = os.path.join(output_dir, filename)

        # Save image
        cv2.imwrite(filepath, frame)
        captured_files.append(filepath)
        images_captured += 1

    result = {
        "success": True,
        "images_captured": images_captured,
        "candidates_found": len(candidate_frames),
        "output_dir": output_dir,
        "captured_files": captured_files,
        "mode": mode,
    }
    if platform_number is not None:
        result["platform_number"] = platform_number

    if show_progress:
        print("\nCapture complete!")
        print(f"Found {len(candidate_frames)} candidate frames")
        print(f"Captured {images_captured} images")
        print(f"Saved to: {output_dir}")

    return result


if __name__ == "__main__":
    # Example usage - modify video_path and platform_number as needed
    result = capture_images_from_video(
        video_path="vid24.MP4",  # Change to your video path
        platform_number=3,  # Change to your platform number
        show_frames=True,  # Set to True to see real-time detection
        show_progress=True,  # Set to True to see progress messages
    )

    import json

    print(json.dumps(result, indent=2))
