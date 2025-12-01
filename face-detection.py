"""
Zipline Video Time Detection

Automatically detects when a zipline rider appears and exits the frame,
returning the start and end timestamps for that segment.
"""

import cv2
import mediapipe as mp
from typing import Optional, Dict, Any, List, Sequence
import os
from pathlib import Path


try:
    from capture_images import capture_images_from_video
except ImportError:
    capture_images_from_video = None


def _add_image_capture_to_result(
    result: dict,
    input_video_path: str,
    segment_start_time: Optional[float],
    segment_end_time: Optional[float],
    capture_images: bool,
    capture_images_mode: Optional[str],
    capture_images_min: Optional[int],
    capture_images_max: Optional[int],
    platform_number: Optional[int],
    direction: str,
    show_progress: bool,
    show_frames: bool,
    capture_offset_after_start: Optional[float],
    capture_offset_before_end: Optional[float],
) -> dict:
    """Helper function to add image capture results to detection result."""
    if (
        capture_images
        and result.get("valid", False)
        and segment_start_time is not None
        and segment_end_time is not None
        and capture_images_from_video is not None
    ):
        # Determine capture mode
        capture_mode = capture_images_mode
        if capture_mode is None:
            # Map direction to capture mode
            if direction == "coming":
                capture_mode = "coming"
            elif direction == "going":
                capture_mode = "going"
            elif direction in ["walking", "kitting", "sitting"]:
                capture_mode = "group"
            else:
                capture_mode = "going"  # Default

        # Capture images from the detected segment
        capture_result = capture_images_from_video(
            video_path=input_video_path,
            mode=capture_mode,
            min_pictures=capture_images_min,
            max_pictures=capture_images_max,
            platform_number=platform_number,
            start_time=segment_start_time,
            end_time=segment_end_time,
            show_progress=show_progress,
            show_frames=show_frames,
        )
        additional_captures = _capture_specific_offsets(
            input_video_path,
            segment_start_time,
            segment_end_time,
            capture_offset_after_start,
            capture_offset_before_end,
            capture_result.get("output_dir")
            if isinstance(capture_result, dict)
            else None,
        )
        if additional_captures:
            if isinstance(capture_result, dict):
                capture_result.setdefault("extra_captures", []).extend(
                    additional_captures
                )
            else:
                capture_result = {
                    "success": True,
                    "captured_files": [],
                    "images_captured": 0,
                    "candidates_found": 0,
                    "output_dir": os.path.join(
                        os.path.dirname(os.path.abspath(input_video_path)),
                        f"{Path(input_video_path).stem}-images",
                    ),
                    "mode": capture_mode,
                    "extra_captures": additional_captures,
                }
        result["image_capture"] = capture_result

    return result


def _capture_specific_offsets(
    video_path: str,
    segment_start_time: Optional[float],
    segment_end_time: Optional[float],
    offset_after_start: Optional[float],
    offset_before_end: Optional[float],
    base_output_dir: Optional[str],
) -> List[str]:
    if segment_start_time is None or segment_end_time is None:
        return []

    safe_after = (
        offset_after_start if offset_after_start and offset_after_start > 0 else None
    )
    safe_before = (
        offset_before_end if offset_before_end and offset_before_end > 0 else None
    )

    if safe_after is None and safe_before is None:
        return []

    output_dir = (
        base_output_dir
        if base_output_dir
        else os.path.join(
            os.path.dirname(os.path.abspath(video_path)),
            f"{Path(video_path).stem}-images",
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    captured_files: List[str] = []
    if safe_after is not None:
        target_time = segment_start_time + safe_after
        if target_time < segment_end_time:
            captured = _capture_frame_at_time(
                video_path, target_time, output_dir, f"after_start_{safe_after:.2f}"
            )
            if captured:
                captured_files.append(captured)

    if safe_before is not None:
        target_time = segment_end_time - safe_before
        if target_time > segment_start_time:
            captured = _capture_frame_at_time(
                video_path, target_time, output_dir, f"before_end_{safe_before:.2f}"
            )
            if captured:
                captured_files.append(captured)

    return captured_files


def _capture_frame_at_time(
    video_path: str,
    time_seconds: float,
    output_dir: str,
    label: str,
) -> Optional[str]:
    if time_seconds < 0:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    filename = f"extra_{label}_t{time_seconds:.2f}s.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath


def is_valid_face_detection(
    bbox,
    keypoints: Optional[Sequence[Any]],
    min_area_ratio: float = 0.01,
    min_aspect_ratio: float = 0.8,
    max_aspect_ratio: float = 2.0,
    min_eye_fraction: float = 0.2,
    max_eye_fraction: float = 0.8,
) -> bool:
    """
    Applies additional heuristics to reduce false positives (e.g., gloves/hands).

    Args:
        bbox: MediaPipe relative bounding box (values between 0 and 1).
        keypoints: Optional relative facial keypoints from MediaPipe.
    """

    if bbox is None:
        return False

    width_ratio = bbox.width
    height_ratio = bbox.height

    if width_ratio <= 0 or height_ratio <= 0:
        return False

    area_ratio = width_ratio * height_ratio
    if area_ratio < min_area_ratio:
        return False

    aspect_ratio = height_ratio / max(width_ratio, 1e-5)
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return False

    if keypoints and len(keypoints) >= 2:
        right_eye = keypoints[0]
        left_eye = keypoints[1]
        if (
            right_eye is None
            or left_eye is None
            or right_eye.x is None
            or left_eye.x is None
        ):
            return False

        eye_distance = abs(left_eye.x - right_eye.x)
        if eye_distance < width_ratio * min_eye_fraction:
            return False
        if eye_distance > width_ratio * max_eye_fraction:
            return False

    return True


def apply_start_offset(
    start_time: Optional[float], offset_seconds: float
) -> Optional[float]:
    """Helper to shift the start time earlier when possible."""
    if start_time is None:
        return None
    if offset_seconds is None or offset_seconds <= 0:
        return start_time
    shift = min(offset_seconds, start_time)
    return max(0.0, start_time - shift)


def trim_video(
    input_video_path: str,
    output_video_path: str,
    start_time: float,
    end_time: float,
) -> bool:
    """
    Trims a video from start_time to end_time and saves it to output_video_path.

    Args:
        input_video_path: Path to the input video file
        output_video_path: Path where the trimmed video will be saved
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        True if successful, False otherwise
    """
    try:
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            cap.release()
            return False

        # Calculate frame numbers for start and end
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Ensure end_frame doesn't exceed total frames
        end_frame = min(end_frame, total_frames)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            return False

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read and write frames from start to end
        frame_count = start_frame
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            frame_count += 1

        # Release resources
        cap.release()
        out.release()

        return True

    except Exception as e:
        print(f"Error trimming video: {str(e)}")
        return False


# Platform-specific configurations
PLATFORM_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        "direction": "going",
        "min_duration": 5.0,
        "max_duration": 10.0,
        "ideal_duration": 8.0,
        "end_trim_seconds": 0.0,
        "backward_extension_seconds": 2.0,
        "start_offset_seconds": 0.0,
        "capture_images": True,
        "capture_images_mode": "going",
        "capture_images_min": 1,
        "capture_images_max": 5,
        "capture_offset_after_start": 2.5,
        "capture_offset_before_end": 1.0,
    },
    2: {
        "direction": "coming",
        "min_duration": 5.0,
        "max_duration": 10.0,
        "ideal_duration": 8.0,
        "end_trim_seconds": 1.0,
        "backward_extension_seconds": 2.0,
        "start_offset_seconds": 0.0,
        "capture_images": True,
        "capture_images_mode": "coming",
        "capture_images_min": 1,
        "capture_images_max": 5,
        "capture_offset_after_start": 2.5,
        "capture_offset_before_end": 2.0,
    },
    3: {
        "direction": "walking",
        "min_duration": 3.0,
        "max_duration": 10.0,
        "ideal_duration": 6.0,
        "end_trim_seconds": 0.0,
        "backward_extension_seconds": 0.0,
        "is_walking": True,
        "start_offset_seconds": 0.0,
        "capture_images": False,
        "capture_images_mode": "group",
        "capture_images_min": 1,
        "capture_images_max": 5,
        "capture_offset_after_start": None,
        "capture_offset_before_end": None,
    },
    4: {
        "direction": "kitting",
        "min_duration": 20.0,
        "max_duration": 60.0,
        "ideal_duration": 30.0,
        "end_trim_seconds": 0.0,
        "backward_extension_seconds": 0.0,
        "start_offset_seconds": 0.0,
        "capture_images": False,
        "capture_images_mode": "group",
        "capture_images_min": 1,
        "capture_images_max": 5,
        "capture_offset_after_start": None,
        "capture_offset_before_end": None,
    },
    5: {
        "direction": "sitting",
        "min_duration": 25.0,
        "max_duration": 60.0,
        "ideal_duration": 40.0,
        "end_trim_seconds": 0.0,
        "backward_extension_seconds": 0.0,
        "start_offset_seconds": 0.0,
        "capture_images": False,
        "capture_images_mode": "group",
        "capture_images_min": 1,
        "capture_images_max": 5,
        "capture_offset_after_start": None,
        "capture_offset_before_end": None,
    },
}


def detect_zipline_segment(
    input_video_path: str,
    direction: Optional[str] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    ideal_duration: Optional[float] = None,
    end_trim_seconds: Optional[float] = None,
    backward_extension_seconds: Optional[float] = None,
    platform_number: Optional[int] = None,
    show_frames: bool = False,
    show_progress: bool = False,
    output_video_path: Optional[str] = None,
    trim_output_path: Optional[str] = None,
    is_walking: bool = False,
    start_offset_seconds: Optional[float] = None,
    capture_images: Optional[bool] = None,
    capture_images_mode: Optional[str] = None,
    capture_images_min: Optional[int] = None,
    capture_images_max: Optional[int] = None,
    capture_offset_after_start: Optional[float] = None,
    capture_offset_before_end: Optional[float] = None,
) -> dict:
    """
    Detects the visible time range of a zipline rider in the video.

    Args:
        input_video_path: Path to the raw video file
        direction: Either "coming", "going", "walking", "kitting", or "sitting" (detection mode)
            - If None and platform_number is provided, uses platform's direction
            - If None and platform_number is not provided, defaults to "coming"
        min_duration: Minimum motion duration to be considered valid (seconds)
            - If None and platform_number is provided, uses platform's min_duration
            - If None and platform_number is not provided, uses default from platform 1
        max_duration: Optional maximum time cap (seconds)
            - If None and platform_number is provided, uses platform's max_duration
            - If None and platform_number is not provided, uses default from platform 1
        ideal_duration: Ideal duration for the clip (seconds)
            - If None and platform_number is provided, uses platform's ideal_duration
            - If None and platform_number is not provided, uses default from platform 1
            - For "going" videos: system picks face detection that gets closest to this duration
        end_trim_seconds: Seconds to remove from end for "coming" videos when end reaches video end
            - If None and platform_number is provided, uses platform's end_trim_seconds
            - If None and platform_number is not provided, defaults to 1.0
            - Only applies to "coming" videos when (start_time + ideal_duration) >= video_duration
        backward_extension_seconds: Seconds to extend backward (earlier) when duration is too short
            - If None and platform_number is provided, uses platform's backward_extension_seconds
            - If None and platform_number is not provided, defaults to 2.0
            - For "coming" videos: used first when duration < min_duration
            - For "going" videos: used as fallback when forward extension fails
        platform_number: Platform number (1, 2, 3, etc.) to use platform-specific settings
            - If provided, overrides direction, min_duration, max_duration, ideal_duration with platform config
            - Platform configs are defined in PLATFORM_CONFIGS dictionary
            - Individual parameters can still override platform settings if explicitly provided
        show_frames: If True, displays frames with detection overlay in real-time (default: False)
        show_progress: If True, displays progress information during processing (default: False)
        output_video_path: Optional path to save video with detection overlay (default: None)
        trim_output_path: Optional path to save trimmed video segment (default: None)
        is_walking: If True, forces walking detection logic regardless of direction/platform
            - If None, automatically generates path: `output_videos/trimmed-{input_filename}`
            - Creates "output_videos" directory in the same directory as input video
            - If provided, uses the specified path
            - Only created if detection is valid
        start_offset_seconds: Optional number of seconds to shift the detected start earlier
            - Example: detected start = 7s, offset = 2s ⇒ final start = 5s (clamped to >= 0)
            - Ignored when <= 0 or when start is already near 0
            - If platform_number is provided and this is None, uses the platform's configured start_offset_seconds
        capture_images: Optional bool to enable image capture within the detected window
            - If None, defers to the platform configuration (default False)
        capture_images_mode: Optional capture mode override ("going", "coming", "group")
            - If None, uses platform configuration, then falls back to detection direction
        capture_images_min / capture_images_max: Optional overrides for number of images to capture
            - If None, use platform configuration when available
        capture_offset_after_start: Optional number of seconds after the detected start
            where an additional frame should be forcibly captured (e.g., 2.0)
        capture_offset_before_end: Optional number of seconds before the detected end
            where another frame should be captured

    Returns:
        dict with:
            - input_video: path to input video
            - direction: detected/requested direction
            - start_time: when subject first enters frame (seconds)
            - end_time: when subject leaves or movement ends (seconds)
            - duration: total duration (seconds)
            - valid: bool indicating if detection meets criteria
            - reason: optional reason if invalid
            - output_video: path to saved video (if output_video_path was provided)
            - trimmed_video: path to trimmed video (if trim_output_path was provided and detection valid)
            - platform_number: platform number used (if provided)

    Platform Configuration:
        Each platform can have its own settings defined in PLATFORM_CONFIGS:
        - direction: "coming" or "going"
        - min_duration: minimum clip duration
        - max_duration: maximum clip duration
        - ideal_duration: ideal clip duration
        - end_trim_seconds: seconds to remove from end for "coming" videos when reaching video end
        - backward_extension_seconds: seconds to extend backward (earlier) when duration is too short
        - start_offset_seconds: seconds to shift detected start earlier (per platform)

    Detection Logic:

    "COMING" Videos:
    - Detects when rider approaches from higher platform to camera position
    - Start: First significant motion detection (rider entering frame, filtered from guide motion)
    - End: start_time + ideal_duration
      * If (start_time + ideal_duration) >= video_duration: end_time = video_duration - end_trim_seconds
        (removes end_trim_seconds to crop out guide's hand switching off camera)
      * If (start_time + ideal_duration) < video_duration: end_time = start_time + ideal_duration
    - Motion detection uses background subtraction to find growing motion patterns
    - Filters out guide's constant small motion, focuses on rider's growing motion
    - Duration rules:
      * If duration > max_duration: Trim from end to reach max_duration
      * If duration < min_duration: Extend forward (later in video) if possible, otherwise use full segment
      * Final clip must always respect min_duration and max_duration constraints

    "GOING" Videos:
    - Detects when rider looks at camera using face detection
    - Collects all face detections throughout the video
    - Picks the face detection that makes total clip closest to ideal_duration
    - If no face detected: uses segment from 0 to ideal_duration (clamped to video length)
    - End time: Always end of video
    - Duration rules:
      * If duration > max_duration: Trim from end to reach ideal_duration (if ideal_duration <= max_duration)
      * If duration < min_duration: Extend forward (backward in time) if possible, otherwise use full segment
      * Final clip must always respect min_duration and max_duration constraints
    "WALKING" Videos:
    - Detects person walking toward camera with start fixed at 0.0s
    - Uses face detections to choose clip end time
      * If video duration <= max_duration: pick detection closest to video end
      * If video duration > max_duration: pick detection closest to max_duration
    - Ensures duration stays between min_duration (>=3s) and max_duration (<=10s)
    - Falls back to full video if constraints cannot be satisfied

    "KITTING" Videos:
    - Detects group of people standing, then one at a time walking toward camera
    - Start: First person movement detected (first person starts walking)
    - End: Last face detection closest to video end (last face we see)
    - Uses motion detection (background subtraction) to detect when first person starts walking
    - Uses face detection to find the last visible face
    - Duration rules:
      * If duration > max_duration: Trim from end to reach max_duration
      * If duration < min_duration: Extend forward (later in video) if possible
      * Final clip must always respect min_duration and max_duration constraints

    "SITTING" Videos:
    - Detects people seated, camera focuses on each person sequentially
    - Start: First face detected (first person in focus)
    - End: Last face detected (closest to video end)
    - Uses face detection only (no motion detection needed)
    - Simple approach: tracks all face detections, uses first and last
    - Duration rules:
      * If duration > max_duration: Trim from end to reach max_duration
      * If duration < min_duration: Extend forward (later in video) if possible
      * Final clip must always respect min_duration and max_duration constraints
    """
    # Apply platform-specific configuration if platform_number is provided
    if platform_number is not None:
        if platform_number not in PLATFORM_CONFIGS:
            return {
                "input_video": input_video_path,
                "direction": direction or "unknown",
                "valid": False,
                "reason": f"Platform {platform_number} not found in PLATFORM_CONFIGS. Available platforms: {list(PLATFORM_CONFIGS.keys())}",
                "platform_number": platform_number,
            }

        platform_config = PLATFORM_CONFIGS[platform_number]

        # Use platform config values if individual parameters are not provided
        if direction is None:
            direction = platform_config.get("direction", "coming")
        if min_duration is None:
            min_duration = platform_config.get("min_duration", 4.0)
        if max_duration is None:
            max_duration = platform_config.get("max_duration", 10.0)
        if ideal_duration is None:
            ideal_duration = platform_config.get("ideal_duration", 8.0)
        if end_trim_seconds is None:
            end_trim_seconds = platform_config.get("end_trim_seconds", 1.0)
        if backward_extension_seconds is None:
            backward_extension_seconds = platform_config.get(
                "backward_extension_seconds", 2.0
            )
        if platform_config.get("is_walking"):
            is_walking = True
        if start_offset_seconds is None:
            start_offset_seconds = platform_config.get("start_offset_seconds", 0.0)
        if capture_images is None:
            capture_images = platform_config.get("capture_images")
        if capture_images_mode is None:
            capture_images_mode = platform_config.get("capture_images_mode")
        if capture_images_min is None:
            capture_images_min = platform_config.get("capture_images_min")
        if capture_images_max is None:
            capture_images_max = platform_config.get("capture_images_max")
        if capture_offset_after_start is None:
            capture_offset_after_start = platform_config.get(
                "capture_offset_after_start"
            )
        if capture_offset_before_end is None:
            capture_offset_before_end = platform_config.get("capture_offset_before_end")
    else:
        # Use defaults if no platform and no explicit values
        if direction is None:
            direction = "coming"
        if min_duration is None:
            min_duration = 4.0
        if max_duration is None:
            max_duration = 10.0
        if ideal_duration is None:
            ideal_duration = 8.0
        if end_trim_seconds is None:
            end_trim_seconds = 1.0
        if backward_extension_seconds is None:
            backward_extension_seconds = 2.0

    if start_offset_seconds is None or start_offset_seconds < 0:
        start_offset_seconds = 0.0
    if capture_images is None:
        capture_images = False
    else:
        capture_images = bool(capture_images)
    capture_offset_after_start = (
        float(capture_offset_after_start)
        if capture_offset_after_start is not None
        else None
    )
    capture_offset_before_end = (
        float(capture_offset_before_end)
        if capture_offset_before_end is not None
        else None
    )

    if is_walking:
        direction = "walking"
    if direction == "walking":
        is_walking = True

    # Auto-generate trim_output_path if not provided
    if trim_output_path is None:
        # Get the directory and filename from input video path
        input_dir = os.path.dirname(os.path.abspath(input_video_path))
        input_filename = os.path.basename(input_video_path)

        # Create output_videos directory in the same directory as input video
        output_videos_dir = os.path.join(input_dir, "output_videos")
        if not os.path.exists(output_videos_dir):
            os.makedirs(output_videos_dir)

        # Generate output filename: trimmed-{original_filename}
        # Preserve the extension
        name, ext = os.path.splitext(input_filename)
        output_filename = f"trimmed-{name}{ext}"
        trim_output_path = os.path.join(output_videos_dir, output_filename)

    # Validate inputs
    if direction not in ["coming", "going", "walking", "kitting", "sitting"]:
        return {
            "input_video": input_video_path,
            "direction": direction,
            "valid": False,
            "reason": f"Invalid direction: {direction}. Must be 'coming', 'going', 'walking', 'kitting', or 'sitting'",
            "platform_number": platform_number,
        }

    try:
        # Open video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            result = {
                "input_video": input_video_path,
                "direction": direction,
                "valid": False,
                "reason": "Could not open video file",
            }
            if platform_number is not None:
                result["platform_number"] = platform_number
            return result

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            result = {
                "input_video": input_video_path,
                "direction": direction,
                "valid": False,
                "reason": "Invalid FPS in video file",
            }
            if platform_number is not None:
                result["platform_number"] = platform_number
            return result

        # Initialize background subtractor for motion detection
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

        # Frame sampling parameters (sample every ~100ms for efficiency)
        sample_interval = max(1, int(fps * 0.1))  # ~10 samples per second

        # Initialize video writer if output path is provided
        video_writer = None
        if output_video_path is not None:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, (frame_width, frame_height)
            )

        if direction == "coming":
            # For "coming": rider comes from higher platform down to lower platform (camera position)
            # Strategy:
            # - Start = first person detection (rider entering frame)
            # - End = last face detection (rider looking at camera)
            # - If face detection period < 2s, extend end by 2s for smoother finish
            # - Apply duration constraints

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0.0

            # Initialize MediaPipe Face Detection
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5,
            )

            # Face detection parameters
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            min_face_width_px = int(frame_width * 0.06)  # 6% of frame width

            # Track detections
            motion_samples = []  # Collect all motion samples to filter out guide
            face_detections = []  # List of face detection times
            last_faces = []
            last_detection_confidence = 0.0
            last_detected_face = False
            last_motion_box = None
            last_motion_area = 0
            last_contours = []

            frame_count = 0

            # First pass: collect all motion and face detection data
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_time = frame_count / fps

                # Sample frames for efficiency
                if frame_count % sample_interval == 0:
                    # Detect person/motion using background subtraction
                    fg_mask = bg_subtractor.apply(frame)

                    # Morphological operations to reduce noise
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                    # Find contours for person detection
                    contours, _ = cv2.findContours(
                        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    has_motion = False
                    motion_area = 0.0
                    last_contours = contours
                    last_motion_box = None

                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)

                        # Threshold for significant movement
                        min_area = (
                            frame.shape[0] * frame.shape[1]
                        ) * 0.01  # 1% of frame

                        if area > min_area:
                            has_motion = True
                            x, y, w, h = cv2.boundingRect(largest_contour)
                            motion_area = float(w * h)
                            last_motion_box = (x, y, w, h)
                            last_motion_area = motion_area

                    # Store motion sample for later analysis
                    motion_samples.append(
                        {
                            "time": frame_time,
                            "has_motion": has_motion,
                            "area": motion_area,
                        }
                    )

                    # Detect faces using MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)

                    detected_face = False
                    last_faces = []
                    last_detection_confidence = 0.0

                    if results.detections:
                        # Select the detection with largest bounding box
                        primary_detection = max(
                            results.detections,
                            key=lambda det: (
                                det.location_data.relative_bounding_box.width
                                * det.location_data.relative_bounding_box.height
                            ),
                        )

                        bbox = primary_detection.location_data.relative_bounding_box
                        confidence = primary_detection.score[0]

                        if not is_valid_face_detection(
                            bbox, primary_detection.location_data.relative_keypoints
                        ):
                            continue

                        # Convert normalized coordinates to pixel coordinates
                        x = int(bbox.xmin * frame_width)
                        y = int(bbox.ymin * frame_height)
                        x = int(bbox.xmin * frame_width)
                        y = int(bbox.ymin * frame_height)
                        w = int(bbox.width * frame_width)
                        h = int(bbox.height * frame_height)

                        # Ensure coordinates are within frame bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame_width - x)
                        h = min(h, frame_height - y)

                        # Check if face is large enough
                        if w >= min_face_width_px and h >= min_face_width_px:
                            if confidence >= 0.5:
                                detected_face = True
                                last_faces.append((x, y, w, h))
                                last_detection_confidence = confidence

                                # Record face detection time
                                face_detections.append(frame_time)

                    last_detected_face = detected_face

                # Create display frame with overlays if needed
                if show_frames or output_video_path:
                    display_frame = frame.copy()

                    # Draw motion/person detection
                    if last_motion_box:
                        x, y, w, h = last_motion_box
                        cv2.rectangle(
                            display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                        )
                        cv2.putText(
                            display_frame,
                            "Person",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                    # Draw face detection
                    for i, (x, y, w, h) in enumerate(last_faces):
                        cv2.rectangle(
                            display_frame, (x, y), (x + w, y + h), (255, 0, 0), 3
                        )
                        if i == 0:
                            cv2.putText(
                                display_frame,
                                f"Face ({last_detection_confidence:.2f})",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 0),
                                2,
                            )

                    # Add overlay info
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_time:.2f}s | Frame: {frame_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Direction: {direction}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    # Note: Start time will be determined after processing all frames
                    cv2.putText(
                        display_frame,
                        f"Faces: {len(face_detections)}",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                # Show frame with overlay
                if show_frames:
                    cv2.imshow("Coming Detection", display_frame)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        cap.release()
                        face_detection.close()
                        if video_writer:
                            video_writer.release()
                        cv2.destroyAllWindows()
                        result = {
                            "input_video": input_video_path,
                            "direction": direction,
                            "valid": False,
                            "reason": "Detection stopped by user",
                        }
                        if platform_number is not None:
                            result["platform_number"] = platform_number
                        return result

                # Write frame to output video if requested
                if output_video_path and video_writer:
                    video_writer.write(display_frame)

                frame_count += 1

            cap.release()
            face_detection.close()  # Clean up MediaPipe resources
            if video_writer:
                video_writer.release()
            if show_frames:
                cv2.destroyAllWindows()

            # Determine start and end times
            # Filter out guide's motion - look for motion that grows over time (rider approaching)
            motion_with_area = [s for s in motion_samples if s["area"] > 0]

            if not motion_with_area:
                # No motion detected - fallback: use last min_duration seconds
                segment_start_time = max(0.0, video_duration - min_duration)
                segment_end_time = video_duration
            else:
                # Find max area to determine thresholds
                max_area = max(s["area"] for s in motion_with_area)

                # Look for motion that grows significantly (rider approaching, not guide)
                # Guide motion stays small and constant, rider motion grows
                growth_window_size = 5  # Check growth over 5 samples (~0.5 seconds)
                min_growth_factor = 2.0  # Area must grow by at least 2x to be rider

                segment_start_time = None

                for i in range(len(motion_samples) - growth_window_size):
                    sample = motion_samples[i]

                    # Check if this sample has motion
                    if sample["area"] == 0:
                        continue

                    # Look ahead to see if motion grows (rider approaching)
                    future_samples = motion_samples[i : i + growth_window_size]
                    future_areas = [s["area"] for s in future_samples if s["area"] > 0]

                    if len(future_areas) >= 2:
                        initial_area = sample["area"]
                        max_future_area = max(future_areas)

                        # Check if motion grows significantly (rider approaching)
                        if max_future_area >= initial_area * min_growth_factor:
                            # This is likely the rider! Start segment here
                            segment_start_time = sample["time"]
                            break

                # If we didn't find growing motion, use first significant motion
                # but filter out very small constant motion (guide)
                if segment_start_time is None:
                    # Use a higher threshold to filter out guide's small motion
                    guide_filter_threshold = (
                        max_area * 0.15
                    )  # Must be at least 15% of max

                    for sample in motion_samples:
                        if sample["area"] >= guide_filter_threshold:
                            segment_start_time = sample["time"]
                            break

                # Determine end time based on ideal_duration
                if segment_start_time is None:
                    # No rider motion detected - fallback: use segment from (video_duration - ideal_duration) to (video_duration - end_trim_seconds)
                    segment_start_time = max(0.0, video_duration - ideal_duration)
                    # If end would be at video end, remove end_trim_seconds to crop guide's hand
                    if segment_start_time + ideal_duration >= video_duration:
                        segment_end_time = max(
                            segment_start_time + 0.1, video_duration - end_trim_seconds
                        )  # Ensure end > start
                    else:
                        segment_end_time = segment_start_time + ideal_duration
                else:
                    # New logic for "coming" videos:
                    # End time = start_time + ideal_duration
                    # BUT if that reaches video end, remove end_trim_seconds to crop guide's hand switching off camera
                    calculated_end_time = segment_start_time + ideal_duration

                    if calculated_end_time >= video_duration:
                        # End time would be at or past video end, so remove end_trim_seconds to crop guide's hand
                        segment_end_time = max(
                            segment_start_time + 0.1, video_duration - end_trim_seconds
                        )  # Ensure end > start
                    else:
                        # End time is before video end, use calculated end time
                        segment_end_time = calculated_end_time

            used_full_video_fallback = False

            segment_start_time = apply_start_offset(
                segment_start_time, start_offset_seconds
            )

            # Calculate initial duration
            segment_start_time = apply_start_offset(
                segment_start_time, start_offset_seconds
            )

            duration = segment_end_time - segment_start_time

            # Apply duration constraints
            if max_duration is not None and duration > max_duration:
                # Trim from the end to reach max_duration
                segment_end_time = segment_start_time + max_duration
                duration = max_duration

            if duration < min_duration:
                # For "coming" videos: First try extending backward (earlier), then forward (later)
                # Step 1: Try extending backward by backward_extension_seconds
                available_backward = segment_start_time

                if available_backward >= backward_extension_seconds:
                    # Can extend backward
                    segment_start_time = max(
                        0.0, segment_start_time - backward_extension_seconds
                    )
                    duration = segment_end_time - segment_start_time

                    # If still too short after backward extension, try forward extension
                    if duration < min_duration:
                        available_forward = video_duration - segment_end_time
                        forward_extension = 2.0  # Extend forward by 2 seconds

                        if available_forward >= forward_extension:
                            # Can extend forward
                            segment_end_time = min(
                                video_duration, segment_end_time + forward_extension
                            )
                            duration = segment_end_time - segment_start_time
                        else:
                            # Can't extend forward enough, extend as much as possible
                            segment_end_time = video_duration
                            duration = segment_end_time - segment_start_time
                else:
                    # Can't extend backward enough, try forward extension
                    available_forward = video_duration - segment_end_time
                    forward_extension = 2.0  # Extend forward by 2 seconds

                    if available_forward >= forward_extension:
                        # Can extend forward
                        segment_end_time = min(
                            video_duration, segment_end_time + forward_extension
                        )
                        duration = segment_end_time - segment_start_time
                    else:
                        # Can't extend forward enough, extend as much as possible
                        segment_end_time = video_duration
                        duration = segment_end_time - segment_start_time

                # If still too short after both extensions, return invalid
                if duration < min_duration:
                    segment_start_time = 0.0
                    segment_end_time = video_duration
                    duration = segment_end_time - segment_start_time
                    used_full_video_fallback = True

            # Final validation
            if duration >= min_duration or used_full_video_fallback:
                if max_duration is not None and duration > max_duration:
                    # Final trim check
                    segment_end_time = segment_start_time + max_duration
                    duration = max_duration

                result = {
                    "input_video": input_video_path,
                    "direction": direction,
                    "start_time": round(segment_start_time, 2),
                    "end_time": round(segment_end_time, 2),
                    "duration": round(duration, 2),
                    "valid": True,
                }
                if used_full_video_fallback:
                    result["fallback"] = "full_video"
                if output_video_path:
                    result["output_video"] = output_video_path
                if platform_number is not None:
                    result["platform_number"] = platform_number

                # Trim video automatically (only if detection is valid and times are set)
                if (
                    result["valid"]
                    and not used_full_video_fallback
                    and segment_start_time is not None
                    and segment_end_time is not None
                ):
                    if trim_video(
                        input_video_path,
                        trim_output_path,
                        segment_start_time,
                        segment_end_time,
                    ):
                        result["trimmed_video"] = trim_output_path
                    else:
                        result["trim_warning"] = "Failed to create trimmed video"

                # Add image capture if requested
                result = _add_image_capture_to_result(
                    result,
                    input_video_path,
                    segment_start_time,
                    segment_end_time,
                    capture_images,
                    capture_images_mode,
                    capture_images_min,
                    capture_images_max,
                    platform_number,
                    direction,
                    show_progress,
                    show_frames,
                    capture_offset_after_start,
                    capture_offset_before_end,
                )

                return result
            else:
                result = {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": f"Detected duration {duration:.2f}s is below minimum {min_duration}s",
                }
                if platform_number is not None:
                    result["platform_number"] = platform_number
                return result

        elif is_walking or direction == "walking":
            direction = "walking"

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0.0

            # Initialize MediaPipe Face Detection
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5,
            )

            # Face detection parameters
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            min_face_width_px = int(frame_width * 0.06)  # 6% of frame width
            min_consecutive_hits = 3  # 3 consecutive detections for stability

            # Collect all face detections throughout the video
            face_detections = []  
            last_faces = []
            last_detection_confidence = 0.0
            last_detected_stable = False
            consecutive_hits = 0
            last_stable_detection_time = None

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_time = frame_count / fps

                if frame_count % sample_interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)

                    detected_stable_frontal = False
                    last_faces = []
                    last_detection_confidence = 0.0

                    if results.detections:
                        primary_detection = max(
                            results.detections,
                            key=lambda det: (
                                det.location_data.relative_bounding_box.width
                                * det.location_data.relative_bounding_box.height
                            ),
                        )

                        bbox = primary_detection.location_data.relative_bounding_box
                        confidence = primary_detection.score[0]

                        if not is_valid_face_detection(
                            bbox, primary_detection.location_data.relative_keypoints
                        ):
                            continue

                        x = int(bbox.xmin * frame_width)
                        y = int(bbox.ymin * frame_height)
                        w = int(bbox.width * frame_width)
                        h = int(bbox.height * frame_height)

                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame_width - x)
                        h = min(h, frame_height - y)

                        if w >= min_face_width_px and h >= min_face_width_px:
                            if confidence >= 0.5:
                                detected_stable_frontal = True
                                last_faces.append((x, y, w, h))
                                last_detection_confidence = confidence

                    last_detected_stable = detected_stable_frontal

                    if detected_stable_frontal:
                        consecutive_hits += 1
                        if consecutive_hits >= min_consecutive_hits:
                            if (
                                last_stable_detection_time is None
                                or (frame_time - last_stable_detection_time) > 0.5
                            ):
                                bbox_tuple = last_faces[0] if last_faces else None
                                face_detections.append(
                                    {
                                        "time": frame_time,
                                        "confidence": last_detection_confidence,
                                        "bbox": bbox_tuple,
                                    }
                                )
                                last_stable_detection_time = frame_time
                    else:
                        consecutive_hits = 0

                if show_frames or output_video_path:
                    display_frame = frame.copy()

                    for i, (x, y, w, h) in enumerate(last_faces):
                        cv2.rectangle(
                            display_frame, (x, y), (x + w, y + h), (255, 0, 0), 3
                        )

                        if last_detected_stable and i == 0:
                            cv2.putText(
                                display_frame,
                                f"Walking Face ({last_detection_confidence:.2f})",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

                    status_color = (0, 255, 0) if last_detected_stable else (0, 0, 255)
                    status_text = "DETECTED" if last_detected_stable else "SEARCHING"
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_time:.2f}s | Frame: {frame_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Direction: walking | Status: {status_text}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        status_color,
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Detections: {len(face_detections)}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                if show_frames:
                    cv2.imshow("Walking Detection", display_frame)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        cap.release()
                        face_detection.close()
                        if video_writer:
                            video_writer.release()
                        cv2.destroyAllWindows()
                        result = {
                            "input_video": input_video_path,
                            "direction": direction,
                            "valid": False,
                            "reason": "Detection stopped by user",
                        }
                        if platform_number is not None:
                            result["platform_number"] = platform_number
                        return result

                if output_video_path and video_writer:
                    video_writer.write(display_frame)

                frame_count += 1

            cap.release()
            face_detection.close()
            if video_writer:
                video_writer.release()
            if show_frames:
                cv2.destroyAllWindows()

            segment_start_time = 0.0
            used_full_video_fallback = False

            target_end_time = video_duration
            if max_duration is not None and target_end_time > max_duration:
                target_end_time = max_duration

            detection_times = [d["time"] for d in face_detections]
            min_requirement = min_duration if min_duration is not None else 0.0
            candidate_times = [t for t in detection_times if t >= min_requirement]

            if candidate_times:
                segment_end_time = min(
                    candidate_times, key=lambda t: abs(t - target_end_time)
                )
            elif detection_times:
                segment_end_time = min(
                    detection_times, key=lambda t: abs(t - target_end_time)
                )
            else:
                segment_end_time = target_end_time

            segment_end_time = max(
                segment_start_time, min(segment_end_time, video_duration)
            )

            if max_duration is not None and segment_end_time > max_duration:
                segment_end_time = max_duration

            duration = segment_end_time - segment_start_time

            if duration < min_requirement:
                if video_duration >= min_requirement:
                    segment_end_time = min(video_duration, min_requirement)
                    duration = segment_end_time
                else:
                    segment_end_time = video_duration
                    duration = segment_end_time
                    used_full_video_fallback = True

            if max_duration is not None and duration > max_duration:
                segment_end_time = max_duration
                duration = max_duration

            if duration <= 0:
                return {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": "Failed to determine walking segment duration",
                }

            result = {
                "input_video": input_video_path,
                "direction": direction,
                "start_time": round(segment_start_time, 2),
                "end_time": round(segment_end_time, 2),
                "duration": round(duration, 2),
                "valid": True,
            }

            if used_full_video_fallback:
                result["fallback"] = "full_video"

            if output_video_path:
                result["output_video"] = output_video_path
            if platform_number is not None:
                result["platform_number"] = platform_number

            if (
                result["valid"]
                and not used_full_video_fallback
                and segment_start_time is not None
                and segment_end_time is not None
            ):
                if trim_video(
                    input_video_path,
                    trim_output_path,
                    segment_start_time,
                    segment_end_time,
                ):
                    result["trimmed_video"] = trim_output_path
                else:
                    result["trim_warning"] = "Failed to create trimmed video"

            # Add image capture if requested
            result = _add_image_capture_to_result(
                result,
                input_video_path,
                segment_start_time,
                segment_end_time,
                capture_images,
                capture_images_mode,
                capture_images_min,
                capture_images_max,
                platform_number,
                direction,
                show_progress,
                show_frames,
                capture_offset_after_start,
                capture_offset_before_end,
            )

            return result
        elif direction == "kitting":
            # For "kitting": group of people standing, then one at a time they walk toward camera
            # Start: First person detected walking (face + significant motion)
            # End: Last person detected (last face that disappears)

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0.0

            # Initialize background subtractor for motion detection
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=True
            )

            # Initialize MediaPipe Face Detection
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5,
            )

            # Initialize MediaPipe Pose for full body detection
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            # Face detection parameters
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            min_face_width_px = int(frame_width * 0.06)  # 6% of frame width

            # Motion detection threshold (for kitting: walking motion)
            frame_area = frame_width * frame_height
            min_motion_area = (
                frame_area * 0.02
            )  # 2% of frame - walking motion threshold

            # Track detections - person presence (face + motion together indicates walking)
            person_detections = []  # List of (time, has_face, has_motion, motion_area) tuples
            face_records = []  # Track detailed face detections for end time

            frame_count = 0
            results = None
            pose_results = None

            # First pass: collect all person detection data
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_time = frame_count / fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces on all frames
                face_results = face_detection.process(rgb_frame)
                has_face = False
                if face_results.detections:
                    # Check if face is valid and large enough
                    for det in face_results.detections:
                        bbox = det.location_data.relative_bounding_box
                        confidence = det.score[0]

                        if not is_valid_face_detection(
                            bbox, det.location_data.relative_keypoints
                        ):
                            continue

                        x = int(bbox.xmin * frame_width)
                        y = int(bbox.ymin * frame_height)
                        w = int(bbox.width * frame_width)
                        h = int(bbox.height * frame_height)

                        # Clamp to frame bounds
                        x = max(0, min(x, frame_width - 1))
                        y = max(0, min(y, frame_height - 1))
                        w = max(0, min(w, frame_width - x))
                        h = max(0, min(h, frame_height - y))

                        if (
                            w >= min_face_width_px
                            and h >= min_face_width_px
                            and confidence >= 0.5
                        ):
                            has_face = True
                            face_records.append(
                                {
                                    "time": frame_time,
                                    "area": float(w * h),
                                    "width": float(w),
                                    "height": float(h),
                                    "center_x": float(x + w / 2),
                                    "center_y": float(y + h / 2),
                                }
                            )
                            break

                # Detect motion on sampled frames
                has_motion = False
                motion_area = 0.0
                if frame_count % sample_interval == 0:
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
                        motion_area = float(area)

                        if motion_area >= min_motion_area:
                            has_motion = True

                # Detect pose (full body) on sampled frames
                has_person_body = False
                if frame_count % sample_interval == 0:
                    pose_results = pose.process(rgb_frame)
                    if pose_results and pose_results.pose_landmarks:
                        # Check if we have enough keypoints to indicate a person
                        # Require at least torso/hip keypoints
                        landmarks = pose_results.pose_landmarks.landmark
                        # Check for key body points (shoulders, hips)
                        if (
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility
                            > 0.5
                            or landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility
                            > 0.5
                            or landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.5
                            or landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility
                            > 0.5
                        ):
                            has_person_body = True
                else:
                    # For non-sampled frames, still process pose for display
                    pose_results = pose.process(rgb_frame)

                # Store person detection: person is "walking" if they have face AND (motion OR body detected)
                # This filters out people just standing (no motion) vs walking (has motion)
                is_person_walking = has_face and (has_motion or has_person_body)

                person_detections.append(
                    {
                        "time": frame_time,
                        "has_face": has_face,
                        "has_motion": has_motion,
                        "has_person_body": has_person_body,
                        "is_walking": is_person_walking,
                        "motion_area": motion_area,
                    }
                )

                results = face_results

                # Get current detection state for display
                current_detection = person_detections[-1] if person_detections else None

                # Create display frame with overlays if needed
                if show_frames or output_video_path:
                    display_frame = frame.copy()

                    # Draw face detection
                    if results and results.detections:
                        for det in results.detections:
                            bbox = det.location_data.relative_bounding_box
                            x = int(bbox.xmin * frame_width)
                            y = int(bbox.ymin * frame_height)
                            w = int(bbox.width * frame_width)
                            h = int(bbox.height * frame_height)
                            cv2.rectangle(
                                display_frame, (x, y), (x + w, y + h), (255, 0, 0), 3
                            )

                    # Draw pose landmarks if detected
                    if pose_results and pose_results.pose_landmarks:
                        mp_drawing = mp.solutions.drawing_utils
                        mp_drawing.draw_landmarks(
                            display_frame,
                            pose_results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                        )

                    # Add overlay info
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_time:.2f}s | Frame: {frame_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Direction: {direction}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    # Show person detection status
                    if current_detection:
                        status = (
                            "WALKING" if current_detection["is_walking"] else "STANDING"
                        )
                        status_color = (
                            (0, 255, 0)
                            if current_detection["is_walking"]
                            else (0, 0, 255)
                        )
                        cv2.putText(
                            display_frame,
                            f"Person: {status} | Face: {current_detection['has_face']} | Motion: {current_detection['has_motion']}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            status_color,
                            2,
                        )
                        cv2.putText(
                            display_frame,
                            f"Faces detected: {len(face_records)}",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                # Show frame with overlay
                if show_frames:
                    cv2.imshow("Kitting Detection", display_frame)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        cap.release()
                        face_detection.close()
                        if video_writer:
                            video_writer.release()
                        cv2.destroyAllWindows()
                        result = {
                            "input_video": input_video_path,
                            "direction": direction,
                            "valid": False,
                            "reason": "Detection stopped by user",
                        }
                        if platform_number is not None:
                            result["platform_number"] = platform_number
                        return result

                # Write frame to output video if requested
                if output_video_path and video_writer:
                    video_writer.write(display_frame)

                frame_count += 1

            cap.release()
            face_detection.close()
            pose.close()
            if video_writer:
                video_writer.release()
            if show_frames:
                cv2.destroyAllWindows()

            # Determine start and end times using person detections
            # Start: First person detected walking (face + motion/body)
            # End: Last person detected (last face that disappears)

            # Find periods where people are walking (sustained person detections)
            walking_periods = []  # List of (start_time, end_time) for each person walking
            current_walking_start = None
            min_walking_duration = (
                0.5  # Person must be detected walking for at least 0.5 seconds
            )
            consecutive_walking_frames = 0
            required_consecutive = int(
                fps * min_walking_duration / sample_interval
            )  # Convert to sample count

            for detection in person_detections:
                if detection["is_walking"]:
                    if current_walking_start is None:
                        # Start of a new walking period
                        current_walking_start = detection["time"]
                        consecutive_walking_frames = 1
                    else:
                        consecutive_walking_frames += 1
                else:
                    if current_walking_start is not None:
                        # End of walking period - check if it was sustained
                        if consecutive_walking_frames >= required_consecutive:
                            # This is a sustained walking period
                            walking_periods.append(
                                (current_walking_start, detection["time"])
                            )
                        current_walking_start = None
                        consecutive_walking_frames = 0

            # Handle case where walking continues to end of video
            if (
                current_walking_start is not None
                and consecutive_walking_frames >= required_consecutive
            ):
                walking_periods.append((current_walking_start, video_duration))

            # Start: First person walking (first walking period)
            if walking_periods:
                segment_start_time = walking_periods[0][0]
            else:
                # No walking periods found - look for first person with face + motion
                for detection in person_detections:
                    if detection["has_face"] and detection["has_motion"]:
                        segment_start_time = detection["time"]
                        break
                else:
                    # Fallback: use first face detection
                    if face_records:
                        segment_start_time = face_records[0]["time"]
                    else:
                        segment_start_time = 0.0

            # Add a small pre-roll so we catch the person before they move too far
            start_padding_seconds = 0.8  # Capture ~1 second before detected walking
            if segment_start_time > 0:
                segment_start_time = max(
                    0.0, segment_start_time - start_padding_seconds
                )

            # End: Last person facing the camera (use strongest face detection near the end)
            if face_records:
                relevant_faces = [
                    face for face in face_records if face["time"] >= segment_start_time
                ]
                if not relevant_faces:
                    relevant_faces = face_records

                # Focus on the final window (e.g., last 1.5s of faces)
                last_face_time = relevant_faces[-1]["time"]
                end_window_seconds = 1.5
                window_faces = [
                    face
                    for face in relevant_faces
                    if face["time"] >= last_face_time - end_window_seconds
                ]
                if not window_faces:
                    window_faces = relevant_faces

                # Pick the face with the largest area in this final window
                chosen_face = max(window_faces, key=lambda face: face["area"])

                end_padding_seconds = 0.4
                segment_end_time = min(
                    video_duration, chosen_face["time"] + end_padding_seconds
                )
            else:
                # No faces detected - use video end
                segment_end_time = video_duration

            segment_start_time = apply_start_offset(
                segment_start_time, start_offset_seconds
            )

            segment_start_time = apply_start_offset(
                segment_start_time, start_offset_seconds
            )

            # Calculate duration
            duration = segment_end_time - segment_start_time

            # Apply duration constraints
            if max_duration is not None and duration > max_duration:
                # Trim from the end to reach max_duration
                segment_end_time = segment_start_time + max_duration
                duration = max_duration

            if duration < min_duration:
                # Try extending forward (later) if possible
                available_forward = video_duration - segment_end_time
                forward_extension = min_duration - duration

                if available_forward >= forward_extension:
                    segment_end_time = min(
                        video_duration, segment_end_time + forward_extension
                    )
                    duration = segment_end_time - segment_start_time
                else:
                    # Extend as much as possible
                    segment_end_time = video_duration
                    duration = segment_end_time - segment_start_time

            # Final validation
            if duration >= min_duration or duration > 0:
                if max_duration is not None and duration > max_duration:
                    segment_end_time = segment_start_time + max_duration
                    duration = max_duration

                result = {
                    "input_video": input_video_path,
                    "direction": direction,
                    "start_time": round(segment_start_time, 2),
                    "end_time": round(segment_end_time, 2),
                    "duration": round(duration, 2),
                    "valid": True,
                }
                if output_video_path:
                    result["output_video"] = output_video_path
                if platform_number is not None:
                    result["platform_number"] = platform_number

                # Trim video automatically
                if (
                    result["valid"]
                    and segment_start_time is not None
                    and segment_end_time is not None
                ):
                    if trim_video(
                        input_video_path,
                        trim_output_path,
                        segment_start_time,
                        segment_end_time,
                    ):
                        result["trimmed_video"] = trim_output_path
                    else:
                        result["trim_warning"] = "Failed to create trimmed video"

                # Add image capture if requested
                result = _add_image_capture_to_result(
                    result,
                    input_video_path,
                    segment_start_time,
                    segment_end_time,
                    capture_images,
                    capture_images_mode,
                    capture_images_min,
                    capture_images_max,
                    platform_number,
                    direction,
                    show_progress,
                    show_frames,
                    capture_offset_after_start,
                    capture_offset_before_end,
                )

                return result
            else:
                result = {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": f"Detected duration {duration:.2f}s is below minimum {min_duration}s",
                }
                if platform_number is not None:
                    result["platform_number"] = platform_number
                return result
        elif direction == "sitting":
            # For "sitting": people seated, camera focuses on each person sequentially
            # Start: First face detected
            # End: Last face detected (closest to video end)

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0.0

            # Initialize MediaPipe Face Detection
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5,
            )

            # Face detection parameters
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            min_face_width_px = int(frame_width * 0.06)  # 6% of frame width

            # Track face detections
            face_detections = []  # List of face detection times
            first_face_time = None
            last_face_time = None

            frame_count = 0
            results = None

            # Process video to detect faces
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_time = frame_count / fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces
                results = face_detection.process(rgb_frame)

                if results.detections:
                    # Select the detection with largest bounding box
                    primary_detection = max(
                        results.detections,
                        key=lambda det: (
                            det.location_data.relative_bounding_box.width
                            * det.location_data.relative_bounding_box.height
                        ),
                    )

                    bbox = primary_detection.location_data.relative_bounding_box
                    confidence = primary_detection.score[0]

                    if not is_valid_face_detection(
                        bbox, primary_detection.location_data.relative_keypoints
                    ):
                        continue

                    # Convert normalized coordinates to pixel coordinates
                    x = int(bbox.xmin * frame_width)
                    y = int(bbox.ymin * frame_height)
                    w = int(bbox.width * frame_width)
                    h = int(bbox.height * frame_height)

                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame_width - x)
                    h = min(h, frame_height - y)

                    # Check if face is large enough
                    if w >= min_face_width_px and h >= min_face_width_px:
                        if confidence >= 0.5:
                            face_detections.append(frame_time)

                            # Track first and last face
                            if first_face_time is None:
                                first_face_time = frame_time
                            last_face_time = frame_time

                # Create display frame with overlays if needed
                if show_frames or output_video_path:
                    display_frame = frame.copy()

                    # Draw face detection
                    if results and results.detections:
                        for det in results.detections:
                            bbox = det.location_data.relative_bounding_box
                            x = int(bbox.xmin * frame_width)
                            y = int(bbox.ymin * frame_height)
                            w = int(bbox.width * frame_width)
                            h = int(bbox.height * frame_height)
                            cv2.rectangle(
                                display_frame, (x, y), (x + w, y + h), (255, 0, 0), 3
                            )

                    # Add overlay info
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_time:.2f}s | Frame: {frame_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Direction: {direction}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Faces detected: {len(face_detections)}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                # Show frame with overlay
                if show_frames:
                    cv2.imshow("Sitting Detection", display_frame)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        cap.release()
                        face_detection.close()
                        if video_writer:
                            video_writer.release()
                        cv2.destroyAllWindows()
                        result = {
                            "input_video": input_video_path,
                            "direction": direction,
                            "valid": False,
                            "reason": "Detection stopped by user",
                        }
                        if platform_number is not None:
                            result["platform_number"] = platform_number
                        return result

                # Write frame to output video if requested
                if output_video_path and video_writer:
                    video_writer.write(display_frame)

                frame_count += 1

            cap.release()
            face_detection.close()
            if video_writer:
                video_writer.release()
            if show_frames:
                cv2.destroyAllWindows()

            # Determine start and end times
            # Start: First face detected
            if first_face_time is None:
                # No face detected - fallback: use video start
                segment_start_time = 0.0
            else:
                segment_start_time = first_face_time

            # End: Last face detected (closest to video end)
            if last_face_time is None:
                # No face detected - fallback: use video end
                segment_end_time = video_duration
            else:
                segment_end_time = last_face_time

            # Calculate duration
            duration = segment_end_time - segment_start_time

            # Apply duration constraints
            if max_duration is not None and duration > max_duration:
                # Trim from the end to reach max_duration
                segment_end_time = segment_start_time + max_duration
                duration = max_duration

            if duration < min_duration:
                # Try extending forward (later) if possible
                available_forward = video_duration - segment_end_time
                forward_extension = min_duration - duration

                if available_forward >= forward_extension:
                    segment_end_time = min(
                        video_duration, segment_end_time + forward_extension
                    )
                    duration = segment_end_time - segment_start_time
                else:
                    # Extend as much as possible
                    segment_end_time = video_duration
                    duration = segment_end_time - segment_start_time

            # Final validation
            if duration >= min_duration or duration > 0:
                if max_duration is not None and duration > max_duration:
                    segment_end_time = segment_start_time + max_duration
                    duration = max_duration

                result = {
                    "input_video": input_video_path,
                    "direction": direction,
                    "start_time": round(segment_start_time, 2),
                    "end_time": round(segment_end_time, 2),
                    "duration": round(duration, 2),
                    "valid": True,
                }
                if output_video_path:
                    result["output_video"] = output_video_path
                if platform_number is not None:
                    result["platform_number"] = platform_number

                # Trim video automatically
                if (
                    result["valid"]
                    and segment_start_time is not None
                    and segment_end_time is not None
                ):
                    if trim_video(
                        input_video_path,
                        trim_output_path,
                        segment_start_time,
                        segment_end_time,
                    ):
                        result["trimmed_video"] = trim_output_path
                    else:
                        result["trim_warning"] = "Failed to create trimmed video"

                # Add image capture if requested
                result = _add_image_capture_to_result(
                    result,
                    input_video_path,
                    segment_start_time,
                    segment_end_time,
                    capture_images,
                    capture_images_mode,
                    capture_images_min,
                    capture_images_max,
                    platform_number,
                    direction,
                    show_progress,
                    show_frames,
                    capture_offset_after_start,
                    capture_offset_before_end,
                )

                return result
            else:
                result = {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": f"Detected duration {duration:.2f}s is below minimum {min_duration}s",
                }
                if platform_number is not None:
                    result["platform_number"] = platform_number
                return result
        else:
            # For "going": detect when rider looks at the camera
            # Strategy: Find all face detections and pick the one that makes the clip closest to min_duration
            # If no face detected, use end - min_duration as start

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0.0

            # Initialize MediaPipe Face Detection
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5,
            )

            # Face detection parameters
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            min_face_width_px = int(frame_width * 0.06)  # 6% of frame width
            min_consecutive_hits = 3  # 3 consecutive detections for stability

            # Collect all face detections throughout the video
            face_detections = []  # List of (time, confidence, bbox) tuples
            last_faces = []
            last_detection_confidence = 0.0
            last_detected_stable = False
            consecutive_hits = 0
            last_stable_detection_time = None

            frame_count = 0

            # First pass: collect all face detections
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_time = frame_count / fps

                # Sample frames for efficiency
                if frame_count % sample_interval == 0:
                    # Convert BGR to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect faces using MediaPipe
                    results = face_detection.process(rgb_frame)

                    # Store detection results
                    detected_stable_frontal = False
                    last_faces = []
                    last_detection_confidence = 0.0

                    if results.detections:
                        # Select the detection with largest bounding box as primary
                        primary_detection = max(
                            results.detections,
                            key=lambda det: (
                                det.location_data.relative_bounding_box.width
                                * det.location_data.relative_bounding_box.height
                            ),
                        )

                        # Get bounding box from MediaPipe detection
                        bbox = primary_detection.location_data.relative_bounding_box
                        confidence = primary_detection.score[0]

                        if not is_valid_face_detection(
                            bbox, primary_detection.location_data.relative_keypoints
                        ):
                            continue

                        # Convert normalized coordinates to pixel coordinates
                        x = int(bbox.xmin * frame_width)
                        y = int(bbox.ymin * frame_height)
                        w = int(bbox.width * frame_width)
                        h = int(bbox.height * frame_height)

                        # Ensure coordinates are within frame bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame_width - x)
                        h = min(h, frame_height - y)

                        # Check if face is large enough (filter out very small detections)
                        if w >= min_face_width_px and h >= min_face_width_px:
                            if confidence >= 0.5:  # MediaPipe confidence threshold
                                detected_stable_frontal = True
                                last_faces.append((x, y, w, h))
                                last_detection_confidence = confidence

                    last_detected_stable = detected_stable_frontal

                    # Track stable detections (consecutive hits)
                    if detected_stable_frontal:
                        consecutive_hits += 1
                        if consecutive_hits >= min_consecutive_hits:
                            # This is a stable detection - record it
                            if (
                                last_stable_detection_time is None
                                or (frame_time - last_stable_detection_time) > 0.5
                            ):  # At least 0.5s apart
                                # Store bbox from last_faces which is already populated
                                bbox_tuple = last_faces[0] if last_faces else None
                                face_detections.append(
                                    {
                                        "time": frame_time,
                                        "confidence": last_detection_confidence,
                                        "bbox": bbox_tuple,
                                    }
                                )
                                last_stable_detection_time = frame_time
                    else:
                        consecutive_hits = 0

                # Create display frame with overlays if needed
                if show_frames or output_video_path:
                    display_frame = frame.copy()

                    # Draw faces from last detection
                    for i, (x, y, w, h) in enumerate(last_faces):
                        cv2.rectangle(
                            display_frame, (x, y), (x + w, y + h), (255, 0, 0), 3
                        )

                        # Add detection label if stable
                        if last_detected_stable and i == 0:  # Only label first face
                            cv2.putText(
                                display_frame,
                                f"Face Detected! ({last_detection_confidence:.2f})",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

                    # Add overlay info
                    status_color = (0, 255, 0) if last_detected_stable else (0, 0, 255)
                    status_text = "DETECTED" if last_detected_stable else "SEARCHING"
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_time:.2f}s | Frame: {frame_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Direction: {direction} | Status: {status_text}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        status_color,
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Detections: {len(face_detections)}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                # Show frame with overlay (with longer delay to reduce lag)
                if show_frames:
                    cv2.imshow("Face Detection", display_frame)
                    # Increased waitKey to reduce lag - waits 30ms instead of 1ms
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        cap.release()
                        face_detection.close()
                        if video_writer:
                            video_writer.release()
                        cv2.destroyAllWindows()
                        result = {
                            "input_video": input_video_path,
                            "direction": direction,
                            "valid": False,
                            "reason": "Detection stopped by user",
                        }
                        if platform_number is not None:
                            result["platform_number"] = platform_number
                        return result

                # Write frame to output video if requested
                if output_video_path and video_writer:
                    video_writer.write(display_frame)

                frame_count += 1

            cap.release()
            face_detection.close()  # Clean up MediaPipe resources
            if video_writer:
                video_writer.release()
            if show_frames:
                cv2.destroyAllWindows()

            # End time is always the end of the video for "going" direction
            segment_end_time = video_duration

            # Determine start time based on face detections
            if len(face_detections) == 0:
                # No face detected: use segment from 0 to ideal_duration (clamped to video length)
                segment_start_time = 0.0
                segment_end_time = min(video_duration, ideal_duration)
            else:
                # Find the face detection that makes the clip closest to ideal_duration
                best_start_time = None
                best_duration_diff = float("inf")

                for detection in face_detections:
                    candidate_start = detection["time"]
                    candidate_duration = segment_end_time - candidate_start

                    # Calculate how close this is to ideal_duration
                    duration_diff = abs(candidate_duration - ideal_duration)

                    if duration_diff < best_duration_diff:
                        best_duration_diff = duration_diff
                        best_start_time = candidate_start

                if best_start_time is not None:
                    segment_start_time = best_start_time
                else:
                    # Fallback: use the earliest detection
                    segment_start_time = face_detections[0]["time"]

            segment_start_time = apply_start_offset(
                segment_start_time, start_offset_seconds
            )

            # Calculate initial duration
            duration = segment_end_time - segment_start_time

            used_full_video_fallback = False

            # Apply duration rules:
            # 1. If duration > max_duration: Trim from end to reach ideal_duration (if ideal_duration <= max_duration)
            # 2. If duration < min_duration: Extend forward (backward in time) if possible, otherwise use full segment

            if max_duration is not None and duration > max_duration:
                # Rule 1: Trim from end to reach ideal_duration (if ideal_duration is within max_duration)
                if ideal_duration <= max_duration:
                    segment_end_time = segment_start_time + ideal_duration
                    duration = ideal_duration
                else:
                    # Ideal exceeds max, so trim to max_duration
                    segment_end_time = segment_start_time + max_duration
                    duration = max_duration

            if duration < min_duration:
                # For "going" videos: First try extending forward (later), then backward (earlier)
                # Step 1: Try extending forward by 2 seconds
                available_forward = video_duration - segment_end_time
                forward_extension = 2.0  # Extend forward by 2 seconds

                if available_forward >= forward_extension:
                    # Can extend forward
                    segment_end_time = min(
                        video_duration, segment_end_time + forward_extension
                    )
                    duration = segment_end_time - segment_start_time

                    # If still too short after forward extension, try backward extension
                    if duration < min_duration:
                        available_backward = segment_start_time

                        if available_backward >= backward_extension_seconds:
                            # Can extend backward
                            segment_start_time = max(
                                0.0, segment_start_time - backward_extension_seconds
                            )
                            duration = segment_end_time - segment_start_time
                        else:
                            # Can't extend backward enough, extend as much as possible
                            segment_start_time = 0.0
                            duration = segment_end_time - segment_start_time
                else:
                    # Can't extend forward enough, try backward extension
                    available_backward = segment_start_time

                    if available_backward >= backward_extension_seconds:
                        # Can extend backward
                        segment_start_time = max(
                            0.0, segment_start_time - backward_extension_seconds
                        )
                        duration = segment_end_time - segment_start_time
                    else:
                        # Can't extend backward enough, extend as much as possible
                        segment_start_time = 0.0
                        duration = segment_end_time - segment_start_time

                # If still too short after both extensions, return invalid
                if duration < min_duration:
                    segment_start_time = 0.0
                    segment_end_time = video_duration
                    duration = segment_end_time - segment_start_time
                    used_full_video_fallback = True

            # Final validation: ensure we respect min and max constraints
            if duration >= min_duration or used_full_video_fallback:
                if max_duration is not None and duration > max_duration:
                    # Final trim to max_duration
                    segment_end_time = segment_start_time + max_duration
                    duration = max_duration

                result = {
                    "input_video": input_video_path,
                    "direction": direction,
                    "start_time": round(segment_start_time, 2),
                    "end_time": round(segment_end_time, 2),
                    "duration": round(duration, 2),
                    "valid": True,
                }
                if used_full_video_fallback:
                    result["fallback"] = "full_video"
                if output_video_path:
                    result["output_video"] = output_video_path
                if platform_number is not None:
                    result["platform_number"] = platform_number

                # Trim video automatically (only if detection is valid and times are set)
                if (
                    result["valid"]
                    and not used_full_video_fallback
                    and segment_start_time is not None
                    and segment_end_time is not None
                ):
                    if trim_video(
                        input_video_path,
                        trim_output_path,
                        segment_start_time,
                        segment_end_time,
                    ):
                        result["trimmed_video"] = trim_output_path
                    else:
                        result["trim_warning"] = "Failed to create trimmed video"

                # Add image capture if requested
                result = _add_image_capture_to_result(
                    result,
                    input_video_path,
                    segment_start_time,
                    segment_end_time,
                    capture_images,
                    capture_images_mode,
                    capture_images_min,
                    capture_images_max,
                    platform_number,
                    direction,
                    show_progress,
                    show_frames,
                    capture_offset_after_start,
                    capture_offset_before_end,
                )

                return result
            else:
                result = {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": f"Detected duration {duration:.2f}s is below minimum {min_duration}s",
                }
                if platform_number is not None:
                    result["platform_number"] = platform_number
                return result

    except Exception as e:
        result = {
            "input_video": input_video_path,
            "direction": direction,
            "valid": False,
            "reason": f"Error processing video: {str(e)}",
        }
        if platform_number is not None:
            result["platform_number"] = platform_number
        return result


if __name__ == "__main__":
    result = detect_zipline_segment(
        input_video_path="going-1.MP4",
        platform_number=1,
        show_frames=True,
    )

    import json

    print(json.dumps(result, indent=2))
