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

# Platform-specific configurations
PLATFORM_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        "mode": "going",
        "min_pictures": 1,
        "max_pictures": 10,
    },
    2: {
        "mode": "coming",
        "min_pictures": 1,
        "max_pictures": 8,
        "guide_region_width_ratio": 0.4,
        "guide_region_height_ratio": 0.8,
    },
    3: {
        "mode": "coming",
        "min_pictures": 3,
        "max_pictures": 6,
        "guide_region_width_ratio": 0.4,
        "guide_region_height_ratio": 0.8,
    },
    4: {
        "mode": "going",
        "min_pictures": 5,
        "max_pictures": 12,
    },
    5: {
        "mode": "coming",
        "min_pictures": 4,
        "max_pictures": 10,
        "guide_region_width_ratio": 0.4,
        "guide_region_height_ratio": 0.8,
    },
    6: {
        "mode": "going",
        "min_pictures": 6,
        "max_pictures": 15,
    },
    7: {
        "mode": "coming",
        "min_pictures": 5,
        "max_pictures": 12,
        "guide_region_width_ratio": 0.4,
        "guide_region_height_ratio": 0.8,
    },
}

DEFAULT_GUIDE_REGION_WIDTH_RATIO = 0.25
DEFAULT_GUIDE_REGION_HEIGHT_RATIO = 0.25


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
) -> Dict[str, Any]:
    """
    Capture images from video based on specified mode and criteria.

    Args:
        video_path: Path to input video file
        mode: Detection mode ("going" or "coming")
            - If None and platform_number is provided, uses platform's mode
            - If None and platform_number is not provided, defaults to "going"
        min_pictures: Minimum number of pictures to capture
            - If None and platform_number is provided, uses platform's min_pictures
            - If None and platform_number is not provided, defaults to 5
        max_pictures: Maximum number of pictures to capture
            - If None and platform_number is provided, uses platform's max_pictures
            - If None and platform_number is not provided, defaults to 10
        platform_number: Platform number (1, 2, 3, etc.) to use platform-specific settings
            - If provided, overrides mode, min_pictures, max_pictures with platform config
            - Platform configs are defined in PLATFORM_CONFIGS dictionary
            - Individual parameters can still override platform settings if explicitly provided
        output_dir: Output directory for images
            - If None, automatically generates: {video_name}-images in same directory as video
            - Example: "vid1.mp4" -> "vid1-images/"
        sharpness_threshold: Minimum sharpness value to accept frame
        show_progress: If True, display progress information
        show_frames: If True, displays frames with detection overlay in real-time
            - For "coming" mode, guide-region overlay uses configured width/height ratios

    Returns:
        Dictionary with capture results (Any type for flexibility)
    """
    guide_region_width_ratio = DEFAULT_GUIDE_REGION_WIDTH_RATIO
    guide_region_height_ratio = DEFAULT_GUIDE_REGION_HEIGHT_RATIO
    # Apply platform-specific configuration if platform_number is provided
    if platform_number is not None:
        if platform_number not in PLATFORM_CONFIGS:
            return {
                "success": False,
                "error": f"Platform {platform_number} not found in PLATFORM_CONFIGS. "
                f"Available platforms: {list(PLATFORM_CONFIGS.keys())}",
                "platform_number": platform_number,
            }

        platform_config = PLATFORM_CONFIGS[platform_number]

        # Use platform config values if individual parameters are not provided
        if mode is None:
            mode = platform_config.get("mode", "going")
        if min_pictures is None:
            min_pictures = platform_config.get("min_pictures", 5)
        if max_pictures is None:
            max_pictures = platform_config.get("max_pictures", 10)
        guide_region_width_ratio = platform_config.get(
            "guide_region_width_ratio", guide_region_width_ratio
        )
        guide_region_height_ratio = platform_config.get(
            "guide_region_height_ratio", guide_region_height_ratio
        )
    else:
        # Use defaults if no platform and no explicit values
        if mode is None:
            mode = "going"
        if min_pictures is None:
            min_pictures = 5
        if max_pictures is None:
            max_pictures = 10

    # Validate mode
    if mode not in ["going", "coming"]:
        return {
            "success": False,
            "error": f"Invalid mode: {mode}. Must be 'going' or 'coming'",
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

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "Could not open video file"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

    # Initialize background subtractor for person detection (coming mode)
    bg_subtractor = None
    if mode == "coming":
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

    # Sample every N frames (process ~10 frames per second for efficiency)
    sample_interval = max(1, int(fps * 0.1))

    # Store candidate frames with scores
    candidate_frames: List[Dict] = []

    frame_count = 0
    images_captured = 0

    if show_progress:
        print(f"Processing video: {video_path}")
        print(f"Mode: {mode}")
        print(f"Target: {min_pictures}-{max_pictures} images")
        print(f"Processing {total_frames} frames...")

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = frame_count / fps

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
                # GOING MODE: Face detection with smile detection
                face_results = face_detection.process(rgb_frame)

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

                        if face_mesh_results.multi_face_landmarks:
                            face_landmarks = face_mesh_results.multi_face_landmarks[0]
                            is_frontal = is_face_looking_at_camera(
                                face_landmarks, frame_width, frame_height
                            )
                            is_smiling, smile_confidence = detect_smile(
                                face_landmarks, frame_width, frame_height
                            )

                        # Only accept if face is looking at camera
                        if is_frontal:
                            # Calculate score: base score from confidence and sharpness
                            # Bonus for smiling
                            base_score = (
                                confidence * 0.5 + (sharpness_score / 500.0) * 0.3
                            )
                            smile_bonus = smile_confidence * 0.2 if is_smiling else 0.0
                            total_score = base_score + smile_bonus

                            candidate_frames.append(
                                {
                                    "frame": frame.copy(),
                                    "frame_count": frame_count,
                                    "time": frame_time,
                                    "score": total_score,
                                    "confidence": confidence,
                                    "sharpness": sharpness_score,
                                    "is_smiling": is_smiling,
                                    "smile_confidence": smile_confidence,
                                    "bbox": (x, y, w, h),
                                }
                            )

                            # Store detection info for visualization
                            last_detection_info = {
                                "bbox": (x, y, w, h),
                                "confidence": confidence,
                                "is_smiling": is_smiling,
                                "smile_confidence": smile_confidence,
                                "sharpness": sharpness_score,
                                "score": total_score,
                                "is_frontal": True,
                            }
                        else:
                            last_detection_info = {
                                "bbox": (x, y, w, h),
                                "confidence": confidence,
                                "is_smiling": False,
                                "smile_confidence": 0.0,
                                "sharpness": sharpness_score,
                                "score": 0.0,
                                "is_frontal": False,
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

        # Create display frame with overlays if needed
        if show_frames:
            display_frame = frame.copy()

            if mode == "going":
                # Draw face detection
                if last_detection_info and last_detection_info.get("is_frontal"):
                    x, y, w, h = last_detection_info["bbox"]
                    confidence = last_detection_info["confidence"]
                    is_smiling = last_detection_info["is_smiling"]
                    smile_conf = last_detection_info["smile_confidence"]
                    sharpness = last_detection_info["sharpness"]
                    score = last_detection_info["score"]

                    # Draw face bounding box
                    color = (0, 255, 0) if is_smiling else (255, 0, 0)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)

                    # Draw labels
                    label = f"Face ({confidence:.2f})"
                    if is_smiling:
                        label += f" | Smile ({smile_conf:.2f})"
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
                        f"Status: {'ACCEPTED' if is_smiling else 'DETECTED'}",
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
                elif last_detection_info:
                    # Face detected but not frontal
                    x, y, w, h = last_detection_info["bbox"]
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        display_frame,
                        "Face (Not Frontal)",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
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

    # Sort candidates by score (highest first)
    candidate_frames.sort(key=lambda x: x["score"], reverse=True)

    # Select frames to capture
    num_to_capture = min(max_pictures, max(min_pictures, len(candidate_frames)))

    if len(candidate_frames) < min_pictures:
        return {
            "success": False,
            "error": f"Only found {len(candidate_frames)} valid frames, "
            f"but {min_pictures} required",
            "candidates_found": len(candidate_frames),
            "output_dir": output_dir,
        }

    # Capture selected frames
    captured_files = []
    for i in range(num_to_capture):
        candidate = candidate_frames[i]
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
        video_path="vid28.MP4",  # Change to your video path
        platform_number=1,  # Change to your platform number
        show_frames=True,  # Set to True to see real-time detection
        show_progress=True,  # Set to True to see progress messages
    )

    import json

    print(json.dumps(result, indent=2))
