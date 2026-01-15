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

# For COMING mode, we want to avoid capturing riders that are still very small /
# far away when no face is detected. We therefore require minimum size and
# position thresholds so we only keep riders that are reasonably close and in
# the expected region of the frame (right‑hand side where the zipline is).
# - COMING_MIN_PERSON_WIDTH_RATIO_NO_FACE: minimum on‑screen width as a
#   fraction of the full frame for a person‑only detection (no face).
# - COMING_MIN_PERSON_AREA_RATIO_NO_FACE: minimum area of the motion blob as a
#   fraction of the full frame (helps reject big noisy blobs that are too thin).
# - COMING_MIN_PERSON_CENTER_X_RATIO: minimum horizontal center position; this
#   makes sure we only consider motion on the rider side of the image and
#   ignore large moving blobs in the valley/trees on the left.
COMING_MIN_PERSON_WIDTH_RATIO_NO_FACE = 0.22  # 22% of frame width - stricter to avoid distant persons
COMING_MIN_PERSON_AREA_RATIO_NO_FACE = 0.045  # 4.5% of frame area - stricter to avoid distant persons
COMING_MIN_PERSON_CENTER_X_RATIO = 0.50       # center must be in right 50% of frame - reasonable
# Minimum relative area - candidate must be at least this % of best candidate's area
COMING_MIN_RELATIVE_AREA_RATIO = 0.70  # 70% of best candidate's area - ensures we pick close persons
# Maximum reasonable person size - if larger, likely an obstruction (hand, etc.)
# Made very lenient - only reject extremely obvious obstructions
COMING_MAX_PERSON_AREA_RATIO = 0.75  # 75% of frame area - if larger, likely obstruction
COMING_MAX_PERSON_WIDTH_RATIO = 0.90  # 90% of frame width - if larger, likely obstruction
# Only check for obstructions near end of video (where guide might cut off)
COMING_OBSTRUCTION_CHECK_END_RATIO = 0.85  # Check in last 15% of video segment (guide's hand appears near end)
COMING_OBSTRUCTION_STRICT_END_RATIO = 0.95  # Apply strictest checks in last 5% of video segment

# For BRIDGE mode (walking on skyline), we still want the person to be reasonably
# visible but we DON'T assume they are on the right-hand side and we don't expect
# guide hands cutting the camera at the end of the clip. Thresholds are therefore:
# - More lenient on minimum size (person can be smaller for most of the walk)
# - No left/right (center_x) restriction – walkway can be in the middle
# NOTE: We will still *completely* ignore any person in the bottom-left guide region
# for bridge, regardless of size or face status (unlike coming mode which has
# exceptions). The guide/waiting area is never a valid capture for bridge.
BRIDGE_MIN_PERSON_WIDTH_RATIO = 0.10   # 10% of frame width - lenient for walking person
BRIDGE_MIN_PERSON_AREA_RATIO = 0.015   # 1.5% of frame area - allow earlier detections as they walk in
BRIDGE_MIN_PERSON_CENTER_X_RATIO = 0.0  # No horizontal bias for bridge – just not in guide region


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


def has_obstruction(
    area: float,
    width: float,
    height: float,
    frame_width: int,
    frame_height: int,
    frame_time: float,
    video_start_time: Optional[float],
    video_end_time: Optional[float],
    max_area_ratio: float = COMING_MAX_PERSON_AREA_RATIO,
    max_width_ratio: float = COMING_MAX_PERSON_WIDTH_RATIO,
    bbox: Optional[Tuple[int, int, int, int]] = None,  # (x, y, w, h) for position-based checks
) -> bool:
    """
    Check if a detected blob is likely an obstruction (hand, object blocking camera)
    rather than a person.
    
    Obstructions are typically:
    - Very large (close to camera, blocking view)
    - Have unusual aspect ratios
    - Cover too much of the frame
    - Occur near the end of video (when guide cuts off)
    - Cover edges/corners of frame (hand reaching to cut camera)
    
    Args:
        area: Area of the detected blob
        width: Width of bounding box
        height: Height of bounding box
        frame_width: Full frame width
        frame_height: Full frame height
        frame_time: Current frame time in video
        video_start_time: Start time of video segment (if None, don't check end ratio)
        video_end_time: End time of video segment (if None, don't check end ratio)
        max_area_ratio: Maximum reasonable area ratio (default: 75%)
        max_width_ratio: Maximum reasonable width ratio (default: 90%)
        bbox: Optional bounding box (x, y, w, h) for position-based checks
    
    Returns:
        True if blob is likely an obstruction, False otherwise
    """
    frame_area = frame_width * frame_height
    area_ratio = area / frame_area
    width_ratio = width / frame_width
    height_ratio = height / frame_height
    
    # Check if we're near the end of the video segment (where guide might cut off)
    is_near_end = False
    is_very_near_end = False
    segment_ratio = 0.0
    if video_start_time is not None and video_end_time is not None:
        segment_duration = video_end_time - video_start_time
        if segment_duration > 0:
            time_in_segment = frame_time - video_start_time
            segment_ratio = time_in_segment / segment_duration
            is_near_end = segment_ratio >= COMING_OBSTRUCTION_CHECK_END_RATIO
            is_very_near_end = segment_ratio >= COMING_OBSTRUCTION_STRICT_END_RATIO
    
    # Always reject if it's EXTREMELY large (regardless of time) - very obvious obstruction
    if area_ratio > max_area_ratio or width_ratio > max_width_ratio:
        return True
    
    # If near end of video (last 15%), apply additional checks for suspicious shapes
    if is_near_end:
        aspect_ratio = height / max(width, 1.0)
        
        # In the last 5%, be very strict - any large blob is suspicious
        if is_very_near_end:
            # If area > 50% in last 5%, likely obstruction (hand covering camera)
            if area_ratio > 0.50:
                return True
            # If width > 70% in last 5%, likely obstruction
            if width_ratio > 0.70:
                return True
        
        # Check aspect ratio - obstructions (like hands) are often more square/compact
        # than a person (which is more elongated)
        # If very square (aspect ratio close to 1) and large, likely obstruction
        if 0.5 < aspect_ratio < 1.5 and area_ratio > 0.40:
            return True
        
        # If height is extremely large relative to frame, likely obstruction
        if height_ratio > 0.90:
            return True
        
        # Check if bbox is near edges/corners (hand reaching to cut camera)
        if bbox is not None:
            x, y, w, h = bbox
            # Check if bbox touches or is very close to frame edges
            edge_threshold = 0.05  # 5% of frame dimension
            touches_left = x < frame_width * edge_threshold
            touches_right = (x + w) > frame_width * (1 - edge_threshold)
            touches_top = y < frame_height * edge_threshold
            touches_bottom = (y + h) > frame_height * (1 - edge_threshold)
            
            # If large blob touches multiple edges, likely obstruction
            if area_ratio > 0.35:
                edges_touched = sum([touches_left, touches_right, touches_top, touches_bottom])
                if edges_touched >= 2:  # Touches 2+ edges
                    return True
                # If touches corner and is large
                if (touches_left or touches_right) and (touches_top or touches_bottom) and area_ratio > 0.30:
                    return True
    
    return False


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
    min_delay_seconds: float = 2.0,
    platform_number: Optional[int] = None,
    output_dir: Optional[str] = None,
    sharpness_threshold: float = 100.0,
    show_progress: bool = False,
    show_frames: bool = False,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    filename_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Capture images from video based on specified mode and criteria.

    Args:
        video_path: Path to input video file
        mode: Detection mode ("going", "coming", "group", or "bridge")
            - "going": Face detection with smile detection and ranking
            - "coming": Person detection filtering out guide in bottom left corner
            - "group": Detects multiple faces and captures frames with most faces visible
            - "bridge": Person walking on skyline towards camera, captures in last 50% of video with face detection
            - If None, defaults to "going"
        min_pictures: Minimum number of pictures to capture (defaults to 5 if None)
        max_pictures: Maximum number of pictures to capture (defaults to 10 if None)
        min_delay_seconds: Minimum delay in seconds between image captures (default: 2.0)
            - After capturing an image at time T, the next image can only be captured at T + min_delay_seconds
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
    # For coming mode: Trim last 1 second from end_time to avoid guide's hand cutting camera
    # This is a safety measure in addition to any trimming done in video detection
    if mode == "coming" and end_time is not None and start_time is not None:
        original_end_time = end_time
        end_time = max(start_time + 0.1, end_time - 1.0)  # Remove 1 second, but ensure end > start
        if show_progress and original_end_time != end_time:
            print(f"Coming mode: Trimmed end_time from {original_end_time:.2f}s to {end_time:.2f}s to avoid guide's hand")
    
    guide_region_width_ratio = DEFAULT_GUIDE_REGION_WIDTH_RATIO
    guide_region_height_ratio = DEFAULT_GUIDE_REGION_HEIGHT_RATIO
    
    # Apply defaults if unspecified
    if mode is None:
        mode = "going"
    if min_pictures is None:
        min_pictures = 5
    if max_pictures is None:
        max_pictures = 10

    # For BRIDGE mode we want a NARROWER guide region so that we only ignore a
    # small strip on the far bottom‑left where people wait, and we don't
    # accidentally treat too much of the bridge as "guide". Coming keeps the
    # wider default (0.4).
    if mode == "bridge":
        guide_region_width_ratio = 0.25  # 25% of frame width for bridge (was 40% default)
    
    # Validate mode
    if mode not in ["going", "coming", "group", "bridge"]:
        return {
            "success": False,
            "error": f"Invalid mode: {mode}. Must be 'going', 'coming', 'group', or 'bridge'",
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
            (lower.startswith("frame_") and lower.endswith(".jpg"))
            or (lower.startswith("extra_after_start_") and lower.endswith(".jpg"))
            or (lower.startswith("extra_before_end_") and lower.endswith(".jpg"))
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
    # (used in coming/group/bridge modes, and as a low-priority fallback for going)
    # Made more sensitive for coming/bridge modes to better detect persons
    bg_subtractor = None
    if mode in ["coming", "group", "going", "bridge"]:
        if mode in ["coming", "bridge"]:
            # More sensitive for coming/bridge modes - lower threshold to detect persons better
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=30, detectShadows=True
            )
        else:
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

        # For BRIDGE mode specifically, skip the very first part of the segment
        # so the background subtractor has time to stabilise. Otherwise we can
        # get a huge "foreground" blob at t=0 that looks like a person and
        # becomes the only (and wrong) candidate.
        if mode == "bridge":
            warmup_start = start_time if start_time is not None else 0.0
            warmup_seconds = 0.7  # ~0.7s of warmup before we accept any candidates
            if frame_time < warmup_start + warmup_seconds:
                frame_count += 1
                continue

        # Skip if before start_time
        if start_time is not None and frame_time < start_time:
            frame_count += 1
            continue

        # Stop if past end_time
        if end_time is not None and frame_time > end_time:
            break

        # For coming mode: Only process frames from the second half of the video segment
        # Person is far away at the start and only gets close towards the end.
        # If segment is 2s-8s (duration 6s), only process from 5s onwards (2s + 6s*0.5).
        if mode == "coming" and start_time is not None and end_time is not None:
            segment_duration = end_time - start_time
            if segment_duration > 0:
                process_from_time = start_time + (segment_duration * 0.5)  # Start from 50% of segment
                if frame_time < process_from_time:
                    frame_count += 1
                    continue

        # For bridge mode: ONLY consider frames in the LAST 5 SECONDS of the video
        # region we are scanning. If start/end are provided, that's the trimmed
        # region; if not, it's the full video [0, video_duration].
        if mode == "bridge":
            lookback_window = 5.0
            region_start = start_time if start_time is not None else 0.0
            region_end = end_time if end_time is not None else video_duration
            region_duration = max(0.0, region_end - region_start)
            if region_duration > 0:
                process_from_time = max(region_start, region_end - lookback_window)
                if frame_time < process_from_time:
                    frame_count += 1
                    continue

        # For visualization, we need to detect faces on every frame 
        # For processing, we sample frames for efficiency
        should_process_frame = frame_count % sample_interval == 0
        should_show_frame = show_frames

        # Convert to RGB if we need to process or show this frame
        if should_process_frame or should_show_frame:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Sample frames for efficiency
        if should_process_frame:
            # Check blurriness first (reject blurry frames early)
            if is_blurry(frame, sharpness_threshold):
                frame_count += 1
                continue

            sharpness_score = calculate_sharpness(frame)
            # Reset detection info only when processing a new frame
            last_detection_info = None

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
                        base_score = confidence * 0.5 + (sharpness_score / 500.0) * 0.25
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

                # Morphological operations to reduce noise - less aggressive for coming mode
                # Use smaller kernel to preserve person shape better
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                # Find contours
                contours, _ = cv2.findContours(
                    fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    # Sort contours by area and try the largest ones
                    # Sometimes the largest might be noise, so we'll check a few
                    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    # Try up to 3 largest contours to find a valid person
                    person_found = False
                    for contour in sorted_contours[:3]:
                        area = cv2.contourArea(contour)
                        
                        # Lower threshold for coming mode - person might be smaller when far away
                        min_area = (frame_width * frame_height) * 0.01  # 1% instead of 2%

                        if area >= min_area:
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Check for faces FIRST to determine if both guide and rider are visible
                            # This allows us to accept frames where person is in guide region BUT rider is also visible
                            face_results = face_detection.process(rgb_frame)
                            total_face_count = len(face_results.detections) if face_results.detections else 0
                            faces_in_guide = 0
                            faces_outside_guide = 0
                            
                            has_face = False
                            has_frontal_face = False
                            face_confidence = 0.0
                            
                            # Process face mesh once for all faces
                            mesh_results = face_mesh.process(rgb_frame) if face_results.detections else None
                            mesh_faces = (
                                list(mesh_results.multi_face_landmarks)
                                if mesh_results and mesh_results.multi_face_landmarks
                                else []
                            )

                            if face_results.detections:
                                # Count faces in guide vs outside guide
                                for idx, det in enumerate(face_results.detections):
                                    bbox = det.location_data.relative_bounding_box
                                    face_x = int(bbox.xmin * frame_width)
                                    face_y = int(bbox.ymin * frame_height)
                                    face_w = int(bbox.width * frame_width)
                                    face_h = int(bbox.height * frame_height)
                                    
                                    face_in_guide = is_in_guide_region(
                                        (face_x, face_y, face_w, face_h),
                                        frame_width,
                                        frame_height,
                                        guide_region_width_ratio,
                                        guide_region_height_ratio,
                                    )
                                    
                                    if face_in_guide:
                                        faces_in_guide += 1
                                    else:
                                        faces_outside_guide += 1
                                        # This face is outside guide region - mark as detected
                                        has_face = True
                                        det_confidence = float(det.score[0])

                                        # Check if this face is frontal using face mesh
                                        if mesh_faces:
                                            lm_index = idx if idx < len(mesh_faces) else 0
                                            try:
                                                if is_face_looking_at_camera(
                                                    mesh_faces[lm_index],
                                                    frame_width,
                                                    frame_height,
                                                ):
                                                    # Found a frontal face - this is the best candidate
                                                    has_frontal_face = True
                                                    face_confidence = det_confidence
                                                    # Don't break - continue checking to see if there's an even better one
                                            except Exception:
                                                # If mesh-based frontal check fails, keep this as non-frontal
                                                if not has_frontal_face:
                                                    face_confidence = max(face_confidence, det_confidence)
                                        else:
                                            # No mesh results, but we have a face
                                            if not has_frontal_face:
                                                face_confidence = max(face_confidence, det_confidence)
                            
                            # Check if person is in guide region
                            person_in_guide = is_in_guide_region(
                                (x, y, w, h),
                                frame_width,
                                frame_height,
                                guide_region_width_ratio,
                                guide_region_height_ratio,
                            )
                            
                            # Calculate person size ratios to determine if someone is close
                            person_width_ratio_temp = w / float(frame_width)
                            frame_area_temp = frame_width * frame_height
                            area_ratio_temp = area / float(frame_area_temp)
                            
                            # Allow if:
                            # 1. Person is outside guide region (normal case - rider detected), OR
                            # 2. Person is in guide region BUT:
                            #    a) 2+ faces detected (both guide and rider visible), OR
                            #    b) Person detection is LARGE (indicating someone is close, even if only 1 face detected)
                            #       - Large = width >= 30% OR area >= 8% (much larger than minimum thresholds)
                            #       - This handles cases where rider is visible but face is too small to detect
                            is_large_person = (person_width_ratio_temp >= 0.30 or area_ratio_temp >= 0.08)
                            allow_capture = not person_in_guide or (person_in_guide and (total_face_count >= 2 or is_large_person))
                            
                            if not allow_capture:
                                # Person is in guide region and it's likely just the guide (no rider visible - only 0-1 faces)
                                rejected_reason = "guide_only"
                                if show_progress:
                                    print(
                                        f"Frame {frame_count} (t={frame_time:.2f}s): REJECTED - {rejected_reason} | "
                                        f"Person in guide region, but only {total_face_count} face(s) detected "
                                        f"(need 2+ to confirm rider is present)"
                                    )
                                last_detection_info = {
                                    "bbox": (x, y, w, h),
                                    "has_face": has_face,
                                    "is_frontal": has_frontal_face,
                                    "sharpness": sharpness_score,
                                    "score": 0.0,
                                    "area": area,
                                    "rejected": rejected_reason,
                                }
                                continue
                            
                            # Person passed guide region check - continue with processing

                            # Before accepting a person‑only detection, make sure
                            # the rider is not still tiny/far away and is in the
                            # expected region of the frame (right‑hand side).
                            # We use on‑screen width, overall area and horizontal
                            # position as proxies for distance/quality.
                            person_width_ratio = w / float(frame_width)
                            frame_area = frame_width * frame_height
                            area_ratio = area / float(frame_area)
                            person_center_x = x + w / 2.0

                            # Check for obstructions (hand, object blocking camera)
                            # Near the end of video, be more aggressive even if face is detected
                            # because guide's hand might be part of the detection
                            is_near_end = False
                            if start_time is not None and end_time is not None:
                                segment_duration = end_time - start_time
                                if segment_duration > 0:
                                    time_in_segment = frame_time - start_time
                                    segment_ratio = time_in_segment / segment_duration
                                    is_near_end = segment_ratio >= COMING_OBSTRUCTION_CHECK_END_RATIO
                            
                            # Check for obstructions:
                            # - Always check if no face detected
                            # - Also check if near end of video (last 15%) even with face, because guide's hand might be in the detection
                            has_obstruction_flag = False
                            if not has_face or is_near_end:
                                # Check for obstructions, passing bbox for position-based checks
                                has_obstruction_flag = has_obstruction(
                                    area, w, h, frame_width, frame_height, 
                                    frame_time, start_time, end_time,
                                    bbox=(x, y, w, h)
                                )
                            # If face is detected and NOT near end, large bounding box is valid (multiple people close together)

                            # Base score from motion and sharpness.
                            area_score = (area / (frame_width * frame_height)) * 0.5
                            sharp_score = (sharpness_score / 500.0) * 0.2

                            # Face bonuses and priority tiers:
                            #   3: Person + frontal face
                            #   2: Person + non-frontal face
                            #   1: Person-only (no face)
                            #   0: Obstruction (hand/object blocking camera) - WORST TIER
                            face_bonus = 0.0
                            priority_level = 1
                            if has_obstruction_flag:
                                priority_level = 0  # Lowest tier for obstructions
                                # Reduce score significantly for obstructions
                                score = area_score * 0.1 + sharp_score * 0.1  # Very low score
                            else:
                                if has_frontal_face:
                                    priority_level = 3
                                    face_bonus = 0.7
                                elif has_face:
                                    priority_level = 2
                                    face_bonus = 0.4
                                score = area_score + sharp_score + face_bonus

                            # Apply size and region checks, but be more lenient when faces are detected.
                            # Face detection is a strong signal that a valid person is present.
                            # When multiple people are close together, the person detection box might be large
                            # and the center_x might be in the left half, but if faces are detected, it's valid.
                            # Reject if:
                            #   - the blob is still very small on screen (even with face), OR
                            #   - the blob sits too far to the left AND no face detected (valley / trees)
                            # Note: Guide region check already done above (allows if both guide+rider visible)
                            
                            # If face is detected, relax the center_x check (person box might encompass both guide and rider)
                            # But still enforce minimum size to avoid very distant persons
                            min_center_x_ratio = COMING_MIN_PERSON_CENTER_X_RATIO
                            if has_face or total_face_count >= 2:
                                # With face detection, allow center_x to be anywhere (person box might be large)
                                min_center_x_ratio = 0.0  # No center_x restriction when faces are present
                            
                            if (
                                person_width_ratio < COMING_MIN_PERSON_WIDTH_RATIO_NO_FACE
                                or area_ratio < COMING_MIN_PERSON_AREA_RATIO_NO_FACE
                                or (person_center_x < frame_width * min_center_x_ratio)
                            ):
                                if person_center_x < frame_width * COMING_MIN_PERSON_CENTER_X_RATIO:
                                    rejected_reason = "wrong_region"
                                elif (
                                    person_width_ratio < COMING_MIN_PERSON_WIDTH_RATIO_NO_FACE
                                    or area_ratio < COMING_MIN_PERSON_AREA_RATIO_NO_FACE
                                ):
                                    rejected_reason = "too_small"
                                else:
                                    rejected_reason = "rejected"
                                
                                # Debug logging to help diagnose why candidates are rejected
                                if show_progress and frame_count % (fps * 2) == 0:  # Every 2 seconds
                                    print(
                                        f"Frame {frame_count} (t={frame_time:.2f}s): REJECTED - {rejected_reason} | "
                                        f"width_ratio={person_width_ratio:.3f} (min={COMING_MIN_PERSON_WIDTH_RATIO_NO_FACE:.3f}), "
                                        f"area_ratio={area_ratio:.3f} (min={COMING_MIN_PERSON_AREA_RATIO_NO_FACE:.3f}), "
                                        f"center_x_ratio={person_center_x/frame_width:.3f} (min={COMING_MIN_PERSON_CENTER_X_RATIO:.3f}), "
                                        f"faces_total={total_face_count}, faces_outside_guide={faces_outside_guide}"
                                    )
                                
                                last_detection_info = {
                                    "bbox": (x, y, w, h),
                                    "has_face": has_face,
                                    "is_frontal": has_frontal_face,
                                    "sharpness": sharpness_score,
                                    "score": score,
                                    "area": area,
                                    "rejected": rejected_reason,
                                }
                                continue

                            candidate_frames.append(
                                {
                                    "frame": frame.copy(),
                                    "frame_count": frame_count,
                                    "time": frame_time,
                                    "score": score,
                                    "priority_level": priority_level,
                                    "sharpness": sharpness_score,
                                    "has_face": has_face,
                                    "is_frontal": has_frontal_face,
                                    "face_confidence": face_confidence,
                                    "bbox": (x, y, w, h),
                                    "area": area,  # Store area for selection prioritization
                                    "has_obstruction": has_obstruction_flag,  # Mark if obstruction detected
                                }
                            )
                            
                            # Debug logging when candidate is accepted
                            if show_progress:
                                obstruction_text = " (OBSTRUCTION)" if has_obstruction_flag else ""
                                print(
                                    f"Frame {frame_count} (t={frame_time:.2f}s): ACCEPTED{obstruction_text} | "
                                    f"Tier={priority_level}, width_ratio={person_width_ratio:.3f}, "
                                    f"area_ratio={area_ratio:.3f}, has_face={has_face}, frontal={has_frontal_face}"
                                )

                            # Store detection info for visualization
                            last_detection_info = {
                                "bbox": (x, y, w, h),
                                "has_face": has_face,
                                "is_frontal": has_frontal_face,
                                "face_confidence": face_confidence,
                                "sharpness": sharpness_score,
                                "score": score,
                                "area": area,
                                "rejected": None,
                                "priority_level": priority_level,
                                "has_obstruction": has_obstruction_flag,
                            }
                            
                            # Found a valid person - break out of contour loop
                            person_found = True
                            break
                    
                    # If no valid person found in any contour, show that we tried
                    if not person_found:
                        # Still set last_detection_info for visualization even if rejected
                        if sorted_contours:
                            largest_contour = sorted_contours[0]
                            largest_area = cv2.contourArea(largest_contour)
                            lx, ly, lw, lh = cv2.boundingRect(largest_contour)
                            min_area = (frame_width * frame_height) * 0.01
                            
                            if show_progress and frame_count % (fps * 2) == 0:
                                area_ratio = largest_area / (frame_width * frame_height)
                                print(
                                    f"Frame {frame_count} (t={frame_time:.2f}s): Person detection failed | "
                                    f"Found {len(sorted_contours)} contours, largest area={largest_area:.0f} ({area_ratio*100:.2f}%), "
                                    f"min_area={min_area:.0f}, bbox=({lx},{ly},{lw},{lh})"
                                )
                            
                            # Store info for visualization (will show as rejected)
                            last_detection_info = {
                                "bbox": (lx, ly, lw, lh),
                                "has_face": False,
                                "is_frontal": False,
                                "sharpness": sharpness_score,
                                "score": 0.0,
                                "area": largest_area,
                                "rejected": "too_small" if largest_area < min_area else "rejected",
                            }

            elif mode == "bridge":
                # BRIDGE MODE: Person walking on skyline towards camera
                # Similar to coming mode but with tier system like going mode
                # Uses same exclusion zone as coming mode (bottom left)
                # Use background subtraction for person detection
                if bg_subtractor is None:
                    frame_count += 1
                    continue
                # Type assertion: bg_subtractor is not None after check
                assert bg_subtractor is not None
                fg_mask = bg_subtractor.apply(frame)

                # Morphological operations to reduce noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                # Find contours
                contours, _ = cv2.findContours(
                    fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    # Sort contours by area and try the largest ones
                    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    # Try up to 3 largest contours to find a valid person
                    person_found = False
                    for contour in sorted_contours[:3]:
                        area = cv2.contourArea(contour)
                        min_area = (frame_width * frame_height) * 0.01  # 1%

                        if area >= min_area:
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Check for faces FIRST to determine if both guide and rider are visible
                            face_results = face_detection.process(rgb_frame)
                            total_face_count = len(face_results.detections) if face_results.detections else 0
                            faces_in_guide = 0
                            faces_outside_guide = 0
                            
                            has_face = False
                            has_frontal_face = False
                            is_smiling = False
                            smile_confidence = 0.0
                            eyes_open = True
                            eyes_open_conf = 0.0
                            face_confidence = 0.0
                            
                            # Process face mesh once for all faces
                            mesh_results = face_mesh.process(rgb_frame) if face_results.detections else None
                            mesh_faces = (
                                list(mesh_results.multi_face_landmarks)
                                if mesh_results and mesh_results.multi_face_landmarks
                                else []
                            )

                            if face_results.detections:
                                # Count faces in guide vs outside guide
                                for idx, det in enumerate(face_results.detections):
                                    bbox = det.location_data.relative_bounding_box
                                    face_x = int(bbox.xmin * frame_width)
                                    face_y = int(bbox.ymin * frame_height)
                                    face_w = int(bbox.width * frame_width)
                                    face_h = int(bbox.height * frame_height)
                                    
                                    # Check if face is large enough (at least 6% of frame width)
                                    min_face_width = int(frame_width * 0.06)
                                    if face_w < min_face_width or face_h < min_face_width:
                                        continue
                                    
                                    face_in_guide = is_in_guide_region(
                                        (face_x, face_y, face_w, face_h),
                                        frame_width,
                                        frame_height,
                                        guide_region_width_ratio,
                                        guide_region_height_ratio,
                                    )
                                    
                                    if face_in_guide:
                                        faces_in_guide += 1
                                    else:
                                        faces_outside_guide += 1
                                        # This face is outside guide region - mark as detected
                                        has_face = True
                                        det_confidence = float(det.score[0])
                                        face_confidence = max(face_confidence, det_confidence)

                                        # Check if this face is frontal using face mesh
                                        if mesh_faces:
                                            lm_index = idx if idx < len(mesh_faces) else 0
                                            try:
                                                if is_face_looking_at_camera(
                                                    mesh_faces[lm_index],
                                                    frame_width,
                                                    frame_height,
                                                ):
                                                    has_frontal_face = True
                                                    # Get smile and eyes for this face
                                                    face_landmarks = mesh_faces[lm_index]
                                                    is_smiling, smile_confidence = detect_smile(
                                                        face_landmarks, frame_width, frame_height
                                                    )
                                                    eyes_open, eyes_open_conf = are_eyes_open(
                                                        face_landmarks, frame_width, frame_height
                                                    )
                                            except Exception:
                                                pass
                            
                            # Check if person is in guide region.
                            # For BRIDGE we NEVER accept anything inside the guide region,
                            # even if it's large or has faces – that area is reserved for
                            # people waiting / next to go.
                            person_in_guide = is_in_guide_region(
                                (x, y, w, h),
                                frame_width,
                                frame_height,
                                guide_region_width_ratio,
                                guide_region_height_ratio,
                            )
                            if person_in_guide:
                                rejected_reason = "guide_only"
                                last_detection_info = {
                                    "bbox": (x, y, w, h),
                                    "has_face": has_face,
                                    "is_frontal": has_frontal_face,
                                    "sharpness": sharpness_score,
                                    "score": 0.0,
                                    "area": area,
                                    "rejected": rejected_reason,
                                }
                                continue
                            
                            # Calculate person size ratios (bridge-specific thresholds)
                            person_width_ratio = w / float(frame_width)
                            frame_area = frame_width * frame_height
                            area_ratio = area / float(frame_area)
                            
                            # Person passed guide region check - continue with processing
                            # For BRIDGE mode we do NOT apply the same obstruction logic as COMING.
                            # There is no guide hand switching off the camera – we simply want
                            # the walking person, even when they are very close.
                            has_obstruction_flag = False
                            
                            # Apply size and region checks (bridge-specific thresholds)
                            person_center_x = x + w / 2.0
                            # For bridge we don't constrain center_x (except for guide region
                            # which we already filtered above). Keep this for symmetry in case
                            # we want to tighten later.
                            min_center_x_ratio = BRIDGE_MIN_PERSON_CENTER_X_RATIO
                            
                            if (
                                person_width_ratio < BRIDGE_MIN_PERSON_WIDTH_RATIO
                                or area_ratio < BRIDGE_MIN_PERSON_AREA_RATIO
                                or (person_center_x < frame_width * min_center_x_ratio)
                            ):
                                last_detection_info = {
                                    "bbox": (x, y, w, h),
                                    "has_face": has_face,
                                    "is_frontal": has_frontal_face,
                                    "sharpness": sharpness_score,
                                    "score": 0.0,
                                    "area": area,
                                    "rejected": "too_small",
                                }
                                continue
                            
                            # Priority tiers for BRIDGE mode (similar to going mode):
                            #   4: Face + frontal + smiling + eyes open
                            #   3: Face + frontal (+/- smile/eyes)
                            #   2: Face (non-frontal) but still valid size
                            #   1: Person-only (no face)
                            #   0: Obstruction
                            if has_obstruction_flag:
                                priority_level = 0
                                score = (area / (frame_width * frame_height)) * 0.1 + (sharpness_score / 500.0) * 0.1
                            else:
                                if has_frontal_face and is_smiling and eyes_open:
                                    priority_level = 4
                                elif has_frontal_face:
                                    priority_level = 3
                                elif has_face:
                                    priority_level = 2
                                else:
                                    priority_level = 1
                                
                                # Calculate score: base score from confidence/sharpness/area
                                if has_face:
                                    base_score = face_confidence * 0.5 + (sharpness_score / 500.0) * 0.25
                                    smile_bonus = smile_confidence * 0.15 if is_smiling else 0.0
                                    eyes_bonus = eyes_open_conf * 0.1 if eyes_open else 0.0
                                    score = base_score + smile_bonus + eyes_bonus
                                else:
                                    # Person-only fallback
                                    area_score = (area / (frame_width * frame_height)) * 0.5
                                    sharp_score = (sharpness_score / 500.0) * 0.3
                                    score = area_score + sharp_score

                            candidate_frames.append(
                                {
                                    "frame": frame.copy(),
                                    "frame_count": frame_count,
                                    "time": frame_time,
                                    "score": score,
                                    "priority_level": priority_level,
                                    "sharpness": sharpness_score,
                                    "has_face": has_face,
                                    "is_frontal": has_frontal_face,
                                    "is_smiling": is_smiling,
                                    "smile_confidence": smile_confidence,
                                    "eyes_open": eyes_open,
                                    "eyes_open_confidence": eyes_open_conf,
                                    "face_confidence": face_confidence,
                                    "bbox": (x, y, w, h),
                                    "area": area,
                                    "has_obstruction": has_obstruction_flag,
                                }
                            )
                            
                            # Store detection info for visualization
                            last_detection_info = {
                                "bbox": (x, y, w, h),
                                "has_face": has_face,
                                "is_frontal": has_frontal_face,
                                "is_smiling": is_smiling,
                                "smile_confidence": smile_confidence,
                                "eyes_open": eyes_open,
                                "eyes_open_confidence": eyes_open_conf,
                                "face_confidence": face_confidence,
                                "sharpness": sharpness_score,
                                "score": score,
                                "area": area,
                                "rejected": None,
                                "priority_level": priority_level,
                                "has_obstruction": has_obstruction_flag,
                            }
                            
                            # Found a valid person - break out of contour loop
                            person_found = True
                            break
                    
                    # If no valid person found
                    if not person_found:
                        if sorted_contours:
                            largest_contour = sorted_contours[0]
                            largest_area = cv2.contourArea(largest_contour)
                            lx, ly, lw, lh = cv2.boundingRect(largest_contour)
                            
                            last_detection_info = {
                                "bbox": (lx, ly, lw, lh),
                                "has_face": False,
                                "is_frontal": False,
                                "sharpness": sharpness_score,
                                "score": 0.0,
                                "area": largest_area,
                                "rejected": "too_small",
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
                # Store ALL detections for visualization 
                # But only count validated faces for scoring
                face_results = face_detection.process(rgb_frame)
                face_count = 0
                face_bboxes = []  # Validated faces for scoring
                all_face_bboxes = []  # All detected faces for visualization 
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

                        # Store ALL detections for visualization
                        all_face_bboxes.append((x, y, w, h))

                        # Check if face is large enough (at least 3% of frame width for groups)
                        # and has sufficient confidence for scoring
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
                    # Include all_face_bboxes for visualization 
                    last_detection_info = {
                        "person_count": person_count,
                        "face_count": face_count,
                        "total_count": total_count,
                        "person_bboxes": person_bboxes,
                        "face_bboxes": face_bboxes,  # Validated faces for scoring
                        "all_face_bboxes": all_face_bboxes,  # All faces for visualization
                        "avg_confidence": avg_confidence,
                        "sharpness": sharpness_score,
                        "score": total_score,
                    }
                else:
                    # Store all detections even if not enough for group requirement
                    # This allows visualization to show all faces 
                    last_detection_info = {
                        "person_count": person_count,
                        "face_count": face_count,
                        "total_count": total_count,
                        "person_bboxes": person_bboxes,
                        "face_bboxes": face_bboxes,
                        "all_face_bboxes": all_face_bboxes,
                        "avg_confidence": 0.0,
                        "sharpness": sharpness_score,
                        "score": 0.0,
                    }

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
                        eyes_conf = last_detection_info.get("eyes_open_confidence", 0.0)

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

                # Draw face detections for visualization (even if not in last_detection_info)
                face_results_viz = face_detection.process(rgb_frame)
                if face_results_viz and face_results_viz.detections:
                    for det in face_results_viz.detections:
                        bbox = det.location_data.relative_bounding_box
                        face_x = int(bbox.xmin * frame_width)
                        face_y = int(bbox.ymin * frame_height)
                        face_w = int(bbox.width * frame_width)
                        face_h = int(bbox.height * frame_height)

                        # Only draw faces outside guide region
                        if not is_in_guide_region(
                            (face_x, face_y, face_w, face_h),
                            frame_width,
                            frame_height,
                            guide_region_width_ratio,
                            guide_region_height_ratio,
                        ):
                            # Check if frontal using face mesh
                            is_frontal_viz = False
                            if face_mesh:
                                mesh_results_viz = face_mesh.process(rgb_frame)
                                if mesh_results_viz and mesh_results_viz.multi_face_landmarks:
                                    try:
                                        is_frontal_viz = is_face_looking_at_camera(
                                            mesh_results_viz.multi_face_landmarks[0],
                                            frame_width,
                                            frame_height,
                                        )
                                    except Exception:
                                        pass

                            # Color: green for frontal, yellow for non-frontal
                            face_color = (0, 255, 0) if is_frontal_viz else (0, 255, 255)
                            cv2.rectangle(
                                display_frame,
                                (face_x, face_y),
                                (face_x + face_w, face_y + face_h),
                                face_color,
                                2,
                            )
                            face_label = "Face (Frontal)" if is_frontal_viz else "Face"
                            cv2.putText(
                                display_frame,
                                face_label,
                                (face_x, face_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                face_color,
                                2,
                            )

                if last_detection_info:
                    x, y, w, h = last_detection_info["bbox"]
                    has_face = last_detection_info["has_face"]
                    is_frontal = last_detection_info.get("is_frontal", False)
                    sharpness = last_detection_info["sharpness"]
                    score = last_detection_info["score"]
                    priority = last_detection_info.get("priority_level", 1)
                    rejected_reason = last_detection_info.get("rejected")

                    # Draw person bounding box
                    color = (0, 255, 0) if rejected_reason is None else (0, 0, 255)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)

                    # Draw label
                    has_obstruction_viz = last_detection_info.get("has_obstruction", False)
                    if rejected_reason == "guide_only":
                        label = "Guide (filtered)"
                    elif rejected_reason == "too_small":
                        label = "Person (too small)"
                    elif rejected_reason == "wrong_region":
                        label = "Person (wrong region)"
                    elif has_obstruction_viz:
                        label = "Person (OBSTRUCTION - Tier 0)"
                        color = (0, 165, 255)  # Orange color for obstructions
                    else:
                        if is_frontal:
                            label = "Person (Frontal Face)"
                        elif has_face:
                            label = "Person (Face)"
                        else:
                            label = "Person (No Face)"
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
                        f"Sharpness: {sharpness:.1f} | Score: {score:.3f} | Tier: {priority}",
                        (
                            "Status: ACCEPTED"
                            if rejected_reason is None
                            else f"Status: FILTERED ({rejected_reason})"
                        ),
                    ]
                    if has_face:
                        info_text.append(
                            f"Face: {'Frontal' if is_frontal else 'Non-frontal'}"
                        )
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

            elif mode == "bridge":
                # Draw person detection and guide region (similar to coming mode)
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

                # Draw face detections for visualization
                face_results_viz = face_detection.process(rgb_frame)
                if face_results_viz and face_results_viz.detections:
                    for det in face_results_viz.detections:
                        bbox = det.location_data.relative_bounding_box
                        face_x = int(bbox.xmin * frame_width)
                        face_y = int(bbox.ymin * frame_height)
                        face_w = int(bbox.width * frame_width)
                        face_h = int(bbox.height * frame_height)

                        # Only draw faces outside guide region
                        if not is_in_guide_region(
                            (face_x, face_y, face_w, face_h),
                            frame_width,
                            frame_height,
                            guide_region_width_ratio,
                            guide_region_height_ratio,
                        ):
                            # Check if frontal using face mesh
                            is_frontal_viz = False
                            is_smiling_viz = False
                            eyes_open_viz = False
                            if face_mesh:
                                mesh_results_viz = face_mesh.process(rgb_frame)
                                if mesh_results_viz and mesh_results_viz.multi_face_landmarks:
                                    try:
                                        face_landmarks_viz = mesh_results_viz.multi_face_landmarks[0]
                                        is_frontal_viz = is_face_looking_at_camera(
                                            face_landmarks_viz,
                                            frame_width,
                                            frame_height,
                                        )
                                        is_smiling_viz, _ = detect_smile(
                                            face_landmarks_viz, frame_width, frame_height
                                        )
                                        eyes_open_viz, _ = are_eyes_open(
                                            face_landmarks_viz, frame_width, frame_height
                                        )
                                    except Exception:
                                        pass

                            # Color: green for frontal+smiling+eyes, yellow for frontal, red for non-frontal
                            if is_frontal_viz and is_smiling_viz and eyes_open_viz:
                                face_color = (0, 255, 0)  # Green - best tier
                            elif is_frontal_viz:
                                face_color = (0, 255, 255)  # Yellow - good tier
                            else:
                                face_color = (0, 165, 255)  # Orange - lower tier
                            cv2.rectangle(
                                display_frame,
                                (face_x, face_y),
                                (face_x + face_w, face_y + face_h),
                                face_color,
                                2,
                            )
                            face_label = "Face"
                            if is_frontal_viz and is_smiling_viz and eyes_open_viz:
                                face_label = "Face (Tier 4)"
                            elif is_frontal_viz:
                                face_label = "Face (Tier 3)"
                            else:
                                face_label = "Face (Tier 2)"
                            cv2.putText(
                                display_frame,
                                face_label,
                                (face_x, face_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                face_color,
                                2,
                            )

                if last_detection_info:
                    x, y, w, h = last_detection_info["bbox"]
                    has_face = last_detection_info.get("has_face", False)
                    is_frontal = last_detection_info.get("is_frontal", False)
                    is_smiling = last_detection_info.get("is_smiling", False)
                    smile_conf = last_detection_info.get("smile_confidence", 0.0)
                    eyes_open = last_detection_info.get("eyes_open", False)
                    eyes_conf = last_detection_info.get("eyes_open_confidence", 0.0)
                    face_conf = last_detection_info.get("face_confidence", 0.0)
                    sharpness = last_detection_info.get("sharpness", 0.0)
                    score = last_detection_info.get("score", 0.0)
                    priority = last_detection_info.get("priority_level", 1)
                    rejected_reason = last_detection_info.get("rejected")

                    # Draw person bounding box
                    color = (0, 255, 0) if rejected_reason is None else (0, 0, 255)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)

                    # Draw label
                    has_obstruction_viz = last_detection_info.get("has_obstruction", False)
                    if rejected_reason == "guide_only":
                        label = "Guide (filtered)"
                    elif rejected_reason == "too_small":
                        label = "Person (too small)"
                    elif has_obstruction_viz:
                        label = "Person (OBSTRUCTION - Tier 0)"
                        color = (0, 165, 255)  # Orange color for obstructions
                    else:
                        if has_face and is_frontal and is_smiling and eyes_open:
                            label = f"Person (Tier 4: Face+Frontal+Smile+Eyes)"
                        elif has_face and is_frontal:
                            label = f"Person (Tier 3: Face+Frontal)"
                        elif has_face:
                            label = f"Person (Tier 2: Face)"
                        else:
                            label = "Person (Tier 1: No Face)"
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
                        f"Sharpness: {sharpness:.1f} | Score: {score:.3f} | Tier: {priority}",
                        (
                            "Status: ACCEPTED"
                            if rejected_reason is None
                            else f"Status: FILTERED ({rejected_reason})"
                        ),
                    ]
                    if has_face:
                        info_text.append(
                            f"Face: {'Frontal' if is_frontal else 'Non-frontal'} "
                            f"(Conf: {face_conf:.2f})"
                        )
                        if is_smiling:
                            info_text.append(f"Smile: {smile_conf:.2f}")
                        if eyes_open:
                            info_text.append(f"Eyes: {eyes_conf:.2f}")
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
                # Detect faces on THIS frame for visualization 
                face_results_viz = face_detection.process(rgb_frame)

                # Draw people from last_detection_info if available
                person_count = 0
                person_bboxes = []
                face_count = 0
                total_count = 0.0
                avg_confidence = 0.0
                sharpness = 0.0
                score = 0.0

                if last_detection_info:
                    person_count = last_detection_info["person_count"]
                    person_bboxes = last_detection_info["person_bboxes"]
                    face_count = last_detection_info["face_count"]
                    total_count = last_detection_info["total_count"]
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

                # Draw ALL detected faces directly from MediaPipe results
                # This ensures we show ALL faces, not just validated ones
                if face_results_viz and face_results_viz.detections:
                    for i, det in enumerate(face_results_viz.detections):
                        bbox = det.location_data.relative_bounding_box
                        x = int(bbox.xmin * frame_width)
                        y = int(bbox.ymin * frame_height)
                        w = int(bbox.width * frame_width)
                        h = int(bbox.height * frame_height)
                        color = (0, 255, 0)  # Green for faces
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
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
                # Show both validated face count and total detected faces
                total_detected_faces = (
                    len(face_results_viz.detections)
                    if face_results_viz and face_results_viz.detections
                    else face_count
                )
                info_text = [
                    f"Time: {frame_time:.2f}s | Frame: {frame_count}",
                    f"Mode: {mode} | Candidates: {len(candidate_frames)}",
                    f"People: {person_count} | Valid Faces: {face_count} | All Faces: {total_detected_faces} | Total: {total_count:.1f}",
                    f"Avg Confidence: {avg_confidence:.2f} | Sharpness: {sharpness:.1f}",
                    f"Score: {score:.3f} | Status: ACCEPTED"
                    if total_count >= 2
                    else f"Score: {score:.3f} | Status: NEEDS MORE",
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

    # Sort candidates by priority, then area (larger = closer = better), then score (highest first).
    # If priority_level is missing (older candidates), treat as 1.
    # For coming mode, area is CRITICAL - we want the closest/biggest person.
    # Make area much more important than tier for coming mode to avoid selecting far-away persons with faces.
    # For bridge mode, we want the BEST face/person shot when the walker is closest:
    #   - Highest tier (face quality) first
    #   - Then largest area (closest to camera)
    #   - Then score as a tie‑breaker
    def sort_key(cand: Dict[str, Any]) -> Tuple[float, int, float]:
        priority = cand.get("priority_level", 1)
        area = cand.get("area", 0.0)
        if area == 0.0:
            # Fallback: calculate from bbox
            bbox = cand.get("bbox")
            if bbox is not None:
                _, _, w, h = bbox
                area = float(w) * float(h)
        score = cand.get("score", 0.0)
        
        if mode == "coming":
            # For coming mode: prioritize area FIRST (larger = closer = better)
            # Then tier, then score. This ensures we pick close persons even if they don't have faces.
            # Use normalized area (as ratio of frame) to make it comparable across videos
            # Square the area_ratio to make area differences much more significant
            # This ensures that even a 20% larger area will dominate over tier differences
            frame_area = frame_width * frame_height if frame_width > 0 and frame_height > 0 else 1.0
            area_ratio = area / frame_area
            area_ratio_squared = area_ratio * area_ratio  # Square to amplify differences
            # Return: (area_ratio_squared, priority, score) - area is MOST important
            return (area_ratio_squared, priority, score)
        elif mode == "bridge":
            # Bridge mode: emphasize highest tier AND closest walker.
            # Use normalized/squared area so being closer strongly dominates within a tier.
            frame_area = frame_width * frame_height if frame_width > 0 and frame_height > 0 else 1.0
            area_ratio = area / frame_area
            area_ratio_squared = area_ratio * area_ratio
            # Return: (priority, area_ratio_squared, score)
            return (priority, area_ratio_squared, score)
        else:
            # For other modes (going, group): prioritize tier first, then score, then area
            return (priority, score, area)
    
    # For coming/bridge modes: Filter out obstructions (Tier 0) BEFORE sorting
    # Obstructions should never be selected, even if they have large area
    if mode in ["coming", "bridge"]:
        candidate_frames = [c for c in candidate_frames if c.get("priority_level", 1) != 0]
        if not candidate_frames:
            # If all candidates were obstructions, we have a problem
            return {
                "success": False,
                "error": "Only found obstructions (hand/object blocking camera) in all candidate frames. No valid person detections.",
            }
    
    candidate_frames.sort(key=sort_key, reverse=True)
    
    # Debug: Show top candidates for coming/bridge modes (always show, not just when show_progress is True)
    if mode in ["coming", "bridge"] and candidate_frames:
        if mode == "coming":
            sort_desc = "Area (largest first), then Tier, then Score"
        else:
            sort_desc = "Tier, Area (closest), then Score"
        print(f"\n=== TOP CANDIDATES (sorted by {sort_desc}) ===")
        for i, cand in enumerate(candidate_frames[:10]):  # Show top 10
            cand_area = cand.get("area", 0.0)
            if cand_area == 0.0:
                bbox = cand.get("bbox")
                if bbox is not None:
                    _, _, w, h = bbox
                    cand_area = float(w) * float(h)
            area_ratio = cand_area / (frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0
            has_face = cand.get("has_face", False)
            is_frontal = cand.get("is_frontal", False)
            is_smiling = cand.get("is_smiling", False)
            eyes_open = cand.get("eyes_open", False)
            print(
                f"#{i+1}: Area={cand_area:.0f} ({area_ratio*100:.1f}%), "
                f"Tier={cand.get('priority_level', 1)}, Time={cand['time']:.2f}s, "
                f"Score={cand.get('score', 0.0):.3f}, "
                f"Face={has_face}, Frontal={is_frontal}"
            )
            if mode == "bridge":
                print(
                    f"     Smile={is_smiling}, Eyes={eyes_open}, "
                    f"Obstruction={cand.get('has_obstruction', False)}"
                )
            else:
                print(f"     Obstruction={cand.get('has_obstruction', False)}")
        print("=" * 60)

    # Smart selection algorithm that prioritizes quality (tier) and size (area) while
    # respecting min_delay_seconds. The algorithm:
    # 1. First tries to select only the highest-tier candidates with largest area
    # 2. If we can't fill max_pictures with highest tier, adds next tier, etc.
    # 3. Always respects min_delay_seconds between any two selected frames
    selected_frames: List[Dict[str, Any]] = []
    
    # Group candidates by priority tier
    candidates_by_tier: Dict[int, List[Dict[str, Any]]] = {}
    for cand in candidate_frames:
        tier = cand.get("priority_level", 1)
        if tier not in candidates_by_tier:
            candidates_by_tier[tier] = []
        candidates_by_tier[tier].append(cand)
    
    # Sort tiers in descending order (highest priority first)
    sorted_tiers = sorted(candidates_by_tier.keys(), reverse=True)
    
    # Find the best candidate to use as reference
    # For coming mode: largest area (closest person) is best (obstructions already filtered out)
    # For bridge/going/group modes: highest tier + highest score is best
    best_candidate = None
    best_area = 0.0
    if candidate_frames:
        if mode == "coming":
            # For coming mode, best = largest area (already sorted first in list, obstructions filtered)
            # candidate_frames[0] is guaranteed to not be an obstruction (Tier 0) because we filtered them out
            best_candidate = candidate_frames[0]
        else:
            # For bridge/going/group modes, best = highest tier + highest score
            # candidate_frames is already sorted by (priority, score, area) descending
            best_candidate = candidate_frames[0]
        best_area = best_candidate.get("area", 0.0)
        if best_area == 0.0:
            bbox = best_candidate.get("bbox")
            if bbox is not None:
                _, _, w, h = bbox
                best_area = float(w) * float(h)
    
    # Filter out candidates that are too small relative to the best candidate
    # This ensures we only select reasonably close persons, not distant ones
    # Only apply to coming mode (bridge mode uses tier/time-based selection)
    min_relative_area = best_area * COMING_MIN_RELATIVE_AREA_RATIO if best_area > 0 else 0.0
    if min_relative_area > 0 and mode == "coming":
        for tier in sorted_tiers:
            filtered_candidates = []
            for cand in candidates_by_tier[tier]:
                cand_area = cand.get("area", 0.0)
                if cand_area == 0.0:
                    bbox = cand.get("bbox")
                    if bbox is not None:
                        _, _, w, h = bbox
                        cand_area = float(w) * float(h)
                # Only keep candidates that are at least 70% of best area (unless it's the best itself)
                # Use frame_count and time to identify the best candidate (can't use == on dicts with numpy arrays)
                is_best = (best_candidate is not None and 
                          cand.get("frame_count") == best_candidate.get("frame_count") and
                          abs(cand.get("time", 0.0) - best_candidate.get("time", 0.0)) < 0.01)
                if cand_area >= min_relative_area or is_best:
                    filtered_candidates.append(cand)
            candidates_by_tier[tier] = filtered_candidates
    
    # For bridge mode: we specifically want the walker when they are CLOSEST,
    # which should be near the END of the detected segment.
    # Restrict candidates to a final time window (e.g. last 1.0s) so we don't
    # accidentally pick a good face that happened earlier in the second half.
    if mode == "bridge" and candidate_frames:
        # Find the latest candidate time
        max_time = max(c["time"] for c in candidate_frames)
        time_window_seconds = 1.0  # look at the last 1 second of candidates

        for tier in sorted_tiers:
            windowed_candidates = [
                c for c in candidates_by_tier[tier]
                if c["time"] >= max_time - time_window_seconds
            ]
            # If a tier loses all candidates in this window, keep the originals
            # so we always have at least something to choose from.
            if windowed_candidates:
                candidates_by_tier[tier] = windowed_candidates
    
    # Within each tier, sort candidates for selection.
    # - Coming: area first (closest person), then score
    # - Bridge: area first (closest walker), then score
    # - Going/Group: score first (best face/group quality), then area
    for tier in sorted_tiers:
        if mode in ["coming", "bridge"]:
            # Coming/bridge: area first (larger = closer = better)
            candidates_by_tier[tier].sort(
                key=lambda c: (c.get("area", 0.0) or (lambda b: float(b[2]) * float(b[3]) if b else 0.0)(c.get("bbox")), c.get("score", 0.0)),
                reverse=True
            )
        else:
            # Going/group: score first (better quality)
            candidates_by_tier[tier].sort(
                key=lambda c: (c.get("score", 0.0), c.get("area", 0.0) or (lambda b: float(b[2]) * float(b[3]) if b else 0.0)(c.get("bbox"))),
                reverse=True
            )
    
    # For coming mode with max_pictures=1, just pick the absolute best candidate
    # (largest area = closest person) regardless of timing - user wants the BEST image
    if mode == "coming" and max_pictures == 1 and candidate_frames:
        # Already sorted by (area_ratio_squared, priority, score) for coming mode - first one is the largest/closest
        best = candidate_frames[0]
        selected_frames.append(best)
        # Always print selected frame info for coming mode (not just when show_progress is True)
        best_area = best.get("area", 0.0)
        if best_area == 0.0:
            bbox = best.get("bbox")
            if bbox is not None:
                _, _, w, h = bbox
                best_area = float(w) * float(h)
        area_ratio = best_area / (frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0
        has_face = best.get("has_face", False)
        is_frontal = best.get("is_frontal", False)
        print(
            f"\n>>> SELECTED BEST FRAME: Tier={best.get('priority_level', 1)}, Time={best['time']:.2f}s, "
            f"Area={best_area:.0f} ({area_ratio*100:.1f}% of frame), "
            f"Face={has_face}, Frontal={is_frontal}, "
            f"Score={best.get('score', 0.0):.3f}, HasObstruction={best.get('has_obstruction', False)}"
        )
    else:
        # For multiple pictures or other modes, respect min_delay_seconds
        # Try to fill selected_frames by iterating through tiers from highest to lowest
        for tier in sorted_tiers:
            tier_candidates = candidates_by_tier[tier]
            
            # For each candidate in this tier (already sorted by area), check if it can be added
            for cand in tier_candidates:
                if len(selected_frames) >= max_pictures:
                    break
                
                candidate_time = cand["time"]
                can_add = True
                
                # Check if this candidate is far enough from all existing captures
                for selected in selected_frames:
                    time_diff = abs(candidate_time - selected["time"])
                    if time_diff < min_delay_seconds:
                        can_add = False
                        break
                
                if can_add:
                    selected_frames.append(cand)
                    if show_progress:
                        cand_area = cand.get("area", 0.0)
                        if cand_area == 0.0:
                            bbox = cand.get("bbox")
                            if bbox is not None:
                                _, _, w, h = bbox
                                cand_area = float(w) * float(h)
                        area_ratio = cand_area / (frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0
                        print(
                            f"SELECTED: Tier={tier}, Time={cand['time']:.2f}s, "
                            f"Area={cand_area:.0f} ({area_ratio*100:.1f}% of frame), "
                            f"Score={cand.get('score', 0.0):.3f}, HasObstruction={cand.get('has_obstruction', False)}"
                        )
            
            if len(selected_frames) >= max_pictures:
                break
    
    # Sort selected frames by time to maintain chronological order in output
    selected_frames.sort(key=lambda x: x["time"])

    if len(selected_frames) < min_pictures:
        error_msg = f"Only found {len(selected_frames)} valid frames with minimum delay of {min_delay_seconds}s, "
        error_msg += f"but {min_pictures} required. "
        if len(candidate_frames) == 0:
            error_msg += "No candidates were found - this may indicate: "
            error_msg += f"(1) Person never reached minimum size (width >= {COMING_MIN_PERSON_WIDTH_RATIO_NO_FACE*100:.0f}%, "
            error_msg += f"area >= {COMING_MIN_PERSON_AREA_RATIO_NO_FACE*100:.1f}%), "
            error_msg += f"(2) Person was always in wrong region (center_x < {COMING_MIN_PERSON_CENTER_X_RATIO*100:.0f}% of frame), "
            error_msg += "or (3) Person was always in guide region. Try running with show_frames=True to see detection status."
        else:
            error_msg += f"Found {len(candidate_frames)} candidates but none met the min_delay_seconds requirement."
        return {
            "success": False,
            "error": error_msg,
            "candidates_found": len(candidate_frames),
            "output_dir": output_dir,
        }

    # Capture selected frames
    captured_files = []
    for candidate in selected_frames:
        frame = candidate["frame"]
        frame_time = candidate["time"]

        # Generate filename with optional prefix to prevent cleanup conflicts
        if filename_prefix:
            filename = f"{filename_prefix}_frame_{candidate['frame_count']:06d}_t{frame_time:.2f}s.jpg"
        else:
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
