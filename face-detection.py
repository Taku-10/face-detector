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
import numpy as np

# Optional: YOLOv8 pose for multi-person kitting
try:
    from ultralytics import YOLO

    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


try:
    
    from .capture_images import capture_images_from_video
except ImportError:
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
    capture_images_min_delay: float,
    platform_number: Optional[int],
    direction: str,
    show_progress: bool,
    show_frames: bool,
    capture_offset_after_start: Optional[float],
    capture_offset_before_end: Optional[float],
    capture_images_output_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None,
) -> dict:
    """Helper function to add image capture results to detection result."""
    # Capture images regardless of whether AI detection succeeded or failed.
    # For most directions we use the detected (trimmed) segment; for WALKING+BRIDGE
    # we intentionally ignore the trim and let the image capture pipeline look at
    # the tail of the FULL video instead (last 5 seconds of the raw file).
    if capture_images and capture_images_from_video is not None:
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

        # Decide the time range we pass into capture_images_from_video:
        # - Normal case: use the detected segment [segment_start_time, segment_end_time]
        # - Special case for WALKING + BRIDGE capture:
        #     Let the capture pipeline itself decide the last 5 seconds of the FULL video
        #     by passing no explicit time range.
        capture_start = segment_start_time
        capture_end = segment_end_time

        if direction == "walking" and capture_mode == "bridge":
            capture_start = None
            capture_end = None

        # DEBUG: Log image capture parameters
        print(f"DEBUG face_detection_core - About to call capture_images_from_video:")
        print(f"  - mode={capture_mode}")
        print(f"  - min_pictures={capture_images_min}, max_pictures={capture_images_max}")
        print(f"  - output_dir={capture_images_output_dir}")
        print(f"  - start_time={capture_start}, end_time={capture_end}")

        # Capture images from the detected segment
        capture_result = capture_images_from_video(
            video_path=input_video_path,
            mode=capture_mode,
            min_pictures=capture_images_min,
            max_pictures=capture_images_max,
            min_delay_seconds=capture_images_min_delay,
            platform_number=platform_number,
            start_time=capture_start,
            end_time=capture_end,
            show_progress=show_progress,
            show_frames=show_frames,
            output_dir=capture_images_output_dir,
            filename_prefix=filename_prefix,
        )

        additional_captures: List[str] = []

        # If we have a structured result, we can respect capture_images_max
        # when deciding whether to take extra offset-based captures.
        if isinstance(capture_result, dict):
            captured_files = capture_result.get("captured_files", []) or []

            if capture_images_max is not None:
                # How many extra images we are allowed to add
                remaining_slots = max(0, capture_images_max - len(captured_files))
            else:
                remaining_slots = None

            # Only capture offset-based images if we have remaining slots
            if remaining_slots is None or remaining_slots > 0:
                print(f"DEBUG face_detection_core - capture_images_from_video returned: {capture_result}")
                additional_captures = _capture_specific_offsets(
                    input_video_path,
                    segment_start_time,
                    segment_end_time,
                    capture_offset_after_start,
                    capture_offset_before_end,
                    capture_result.get("output_dir"),
                    filename_prefix,
                    remaining_slots,
                )

            extra_files = additional_captures or []

            # Recompute images_captured as main + extras (both are <= max if set)
            total_captured = len(captured_files) + len(extra_files)
            capture_result["images_captured"] = total_captured

            if extra_files:
                capture_result.setdefault("extra_captures", []).extend(extra_files)

        elif additional_captures:
            # No structured capture_result (should not usually happen), synthesize one
            capture_result = {
                "success": True,
                "captured_files": [],
                "images_captured": len(additional_captures),
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
    filename_prefix: Optional[str] = None,
    max_captures: Optional[int] = None,
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

    # Helper to check if we have reached the maximum number of extra captures
    def can_capture_more() -> bool:
        return max_captures is None or len(captured_files) < max_captures

    if safe_after is not None:
        if not can_capture_more():
            return captured_files
        target_time = segment_start_time + safe_after
        if target_time < segment_end_time:
            captured = _capture_frame_at_time(video_path, target_time, output_dir, f"after_start_{safe_after:.2f}", filename_prefix)
            if captured:
                captured_files.append(captured)

    if safe_before is not None:
        if not can_capture_more():
            return captured_files
        target_time = segment_end_time - safe_before
        if target_time > segment_start_time:
            captured = _capture_frame_at_time(video_path, target_time, output_dir, f"before_end_{safe_before:.2f}", filename_prefix)
            if captured:
                captured_files.append(captured)

    return captured_files


def _capture_frame_at_time(
    video_path: str,
    time_seconds: float,
    output_dir: str,
    label: str,
    filename_prefix: Optional[str] = None,
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

    # Add prefix to filename to prevent cleanup conflicts between scenes
    if filename_prefix:
        filename = f"{filename_prefix}_extra_{label}_t{time_seconds:.2f}s.jpg"
    else:
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
        "capture_images_max": 2,
        "capture_images_min_delay": 2.0,
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
        "capture_images_max": 1,
        "capture_images_min_delay": 2.0,
        "capture_offset_after_start": 2.5,
        "capture_offset_before_end": 2.5,
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
        "capture_images": True,
        "capture_images_mode": "bridge",
        "capture_images_min": 1,
        "capture_images_max": 1,
        "capture_images_min_delay": 2.0,
        "capture_offset_after_start": 1.5,
        "capture_offset_before_end": 2,
    },
    4: {
        "direction": "kitting",
        "min_duration": 20.0,
        "max_duration": 60.0,
        "ideal_duration": 30.0,
        "end_trim_seconds": 0.0,
        "backward_extension_seconds": 0.0,
        "start_offset_seconds": 0.0,
        "capture_images": True,
        "capture_images_mode": "group",
        "capture_images_min": 1,
        "capture_images_max": 5,
        "capture_images_min_delay": 2.0,
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
        "capture_images": True,
        "capture_images_mode": "group",
        "capture_images_min": 1,
        "capture_images_max": 5,
        "capture_images_min_delay": 2.0,
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
    capture_images_min_delay: Optional[float] = None,
    capture_offset_after_start: Optional[float] = None,
    capture_offset_before_end: Optional[float] = None,
    capture_images_output_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None,
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
        capture_images_mode: Optional capture mode override ("going", "coming", "group", "bridge")
            - If None, uses platform configuration, then falls back to detection direction
        capture_images_min / capture_images_max: Optional overrides for number of images to capture
            - If None, use platform configuration when available
        capture_images_min_delay: Optional minimum delay in seconds between image captures
            - If None, use platform configuration when available (default: 2.0)
            - After capturing an image at time T, the next image can only be captured at T + delay
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
        if capture_images_min_delay is None:
            capture_images_min_delay = platform_config.get(
                "capture_images_min_delay", 2.0
            )
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
    if capture_images_min_delay is None:
        capture_images_min_delay = 2.0
    else:
        capture_images_min_delay = float(capture_images_min_delay)
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

                # At this point, segment_start_time is based purely on motion.
                # For "coming" videos where the rider starts very far away, this can
                # still be too early (rider not really visible yet). We have
                # face_detections collected, so we can refine the start time to be
                # closer to when a clear face first appears.
                if segment_start_time is not None and face_detections:
                    first_face_time = face_detections[0]

                    # Allow the clip to start up to this many seconds *before*
                    # the first face, so we still see the rider entering while
                    # avoiding very early frames where they're not in view.
                    # We use a slightly longer window for "coming" videos so
                    # we capture more of the ride instead of only the very end.
                    max_lead_before_face = 3.0  # seconds

                    if segment_start_time < first_face_time - max_lead_before_face:
                        segment_start_time = max(
                            0.0, first_face_time - max_lead_before_face
                        )

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
                    capture_images_min_delay,
                    platform_number,
                    direction,
                    show_progress,
                    show_frames,
                    capture_offset_after_start,
                    capture_offset_before_end,
                    capture_images_output_dir,
                    filename_prefix,
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
                capture_images_min_delay,
                platform_number,
                direction,
                show_progress,
                show_frames,
                capture_offset_after_start,
                capture_offset_before_end,
                capture_images_output_dir,
                filename_prefix,
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

            # Initialize YOLOv8 pose for multi-person detection (kitting mode only)
            if not HAS_YOLO:
                return {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": "YOLOv8 (ultralytics) is required for kitting mode but not installed. Install with: pip install ultralytics",
                    "platform_number": platform_number,
                }

            try:
                yolo_pose_model = YOLO("yolov8n-pose.pt")
                print("[kitting] Using YOLOv8 pose (multi-person).")
            except Exception as e:
                return {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": f"Failed to load YOLOv8 pose model: {e}",
                    "platform_number": platform_number,
                }

            # Face detection parameters
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            min_face_width_px = int(frame_width * 0.06)  # 6% of frame width

            # Motion detection threshold (for kitting: walking motion)
            frame_area = frame_width * frame_height
            # Lower threshold further to reliably pick up smaller / distant movers
            # so that even people starting far back in the frame contribute motion.
            # 0.2% of the frame tends to work better across kitting videos.
            min_motion_area = (
                frame_area * 0.002
            )  # 0.2% of frame - walking motion threshold

            # Track detections - person presence (face + motion together indicates walking)
            person_detections = []  # List of (time, has_face, has_motion, motion_area) tuples
            face_records = []  # Track detailed face detections for end time

            frame_count = 0
            results = None
            pose_results = None

            # Per-person tracking state for kitting (ID -> state dict)
            person_states: Dict[int, Dict[str, Any]] = {}

            # Movement thresholds (tunable)
            movement_threshold_px = max(
                5.0, frame_width * 0.01
            )  # how many pixels = “moved”
            stationary_required_seconds = 0.7  # how long someone must be still first
            stationary_required_frames = int(stationary_required_seconds * fps)

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
                face_count_valid = 0
                face_boxes_valid: list[tuple[int, int, int, int]] = []
                if face_results.detections:
                    # Check all faces (not just one) so we can handle multiple people simultaneously
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
                            face_count_valid += 1
                            face_boxes_valid.append((x, y, w, h))
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

                # Detect motion on sampled frames
                has_motion = False
                motion_area = 0.0
                motion_count = 0
                person_bboxes: list[tuple[int, int, int, int]] = []
                if frame_count % sample_interval == 0:
                    fg_mask = bg_subtractor.apply(frame)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                    contours, _ = cv2.findContours(
                        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        # Count all sufficiently large motion blobs to allow multiple people
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area >= min_motion_area:
                                x, y, w, h = cv2.boundingRect(contour)
                                person_bboxes.append((x, y, w, h))
                                motion_count += 1
                                motion_area = max(motion_area, float(area))

                        if motion_count > 0:
                            has_motion = True

                # Detect & TRACK people with YOLOv8 pose (gives IDs + keypoints)
                has_person_body = False
                pose_overlays = []
                yolo_pose_count = 0

                try:
                    # Use tracking so each person gets a stable ID across frames
                    yolo_results = yolo_pose_model.track(
                        frame,
                        persist=True,
                        verbose=False,
                        classes=[0],  # class 0 = person
                    )

                    if yolo_results and len(yolo_results) > 0:
                        r = yolo_results[0]
                        boxes = r.boxes
                        kps_tensor = r.keypoints.xy if r.keypoints is not None else None

                        for idx, box in enumerate(boxes):
                            if box.id is None:
                                continue

                            person_id = int(box.id)
                            cx, cy, w, h = box.xywh[0].tolist()

                            # Update per-person state
                            state = person_states.setdefault(
                                person_id,
                                {
                                    "last_pos": (cx, cy),
                                    "stationary_frames": 0,
                                    "walking": False,
                                    "walking_start_time": None,
                                },
                            )

                            px, py = state["last_pos"]
                            dist = float(np.hypot(cx - px, cy - py))

                            if dist < movement_threshold_px:
                                # Still essentially stationary
                                state["stationary_frames"] += 1
                            else:
                                # Person moved noticeably
                                if (
                                    not state["walking"]
                                    and state["stationary_frames"]
                                    >= stationary_required_frames
                                ):
                                    # This is the FIRST time this person goes from stationary → walking
                                    state["walking"] = True
                                    state["walking_start_time"] = frame_time
                                    print(
                                        f"[kitting] person_id={person_id} started walking at "
                                        f"{frame_time:.2f}s (stationary_frames={state['stationary_frames']}, "
                                        f"dist={dist:.2f})"
                                    )

                                # Reset stationary counter after movement
                                state["stationary_frames"] = 0

                            state["last_pos"] = (cx, cy)

                            # Pose overlay for display (optional)
                            if kps_tensor is not None and idx < len(kps_tensor):
                                kps_np = kps_tensor[idx].cpu().numpy()
                                if kps_np.size > 0:
                                    yolo_pose_count += 1
                                    has_person_body = True
                                    pose_overlays.append(
                                        {"kps": kps_np, "normalized": False}
                                    )

                    # print(f"[kitting] frame {frame_count}: tracked people={len(person_states)}")
                except Exception as e:
                    print(f"[kitting] YOLO pose tracking failed: {e}")

                # Store person detection: allow multiple simultaneous people
                # Count walking as any valid face plus motion/body; track how many faces/motion blobs we saw
                walking_count = max(face_count_valid, motion_count, yolo_pose_count)
                is_person_walking = walking_count > 0 and (
                    has_motion or has_person_body or has_face
                )

                person_detections.append(
                    {
                        "time": frame_time,
                        "has_face": has_face,
                        "face_count": face_count_valid,
                        "has_motion": has_motion,
                        "motion_count": motion_count,
                        "has_person_body": has_person_body,
                        "is_walking": is_person_walking,
                        "walking_count": walking_count,
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

                    # Draw YOLO pose skeletons for all detected people
                    if pose_overlays:
                        # YOLO skeleton connections (COCO format - 17 keypoints)
                        yolo_skeleton = [
                            (0, 1),  # nose to left_eye
                            (0, 2),  # nose to right_eye
                            (1, 3),  # left_eye to left_ear
                            (2, 4),  # right_eye to right_ear
                            (5, 6),  # left_shoulder to right_shoulder
                            (5, 7),  # left_shoulder to left_elbow
                            (7, 9),  # left_elbow to left_wrist
                            (6, 8),  # right_shoulder to right_elbow
                            (8, 10),  # right_elbow to right_wrist
                            (5, 11),  # left_shoulder to left_hip
                            (6, 12),  # right_shoulder to right_hip
                            (11, 12),  # left_hip to right_hip
                            (11, 13),  # left_hip to left_knee
                            (13, 15),  # left_knee to left_ankle
                            (12, 14),  # right_hip to right_knee
                            (14, 16),  # right_knee to right_ankle
                        ]

                        for overlay in pose_overlays:
                            kps = overlay["kps"]
                            # Draw skeleton lines
                            for a, b in yolo_skeleton:
                                if a < len(kps) and b < len(kps):
                                    pt1 = (int(kps[a][0]), int(kps[a][1]))
                                    pt2 = (int(kps[b][0]), int(kps[b][1]))
                                    # Only draw if both points are valid (non-zero)
                                    if (
                                        pt1[0] > 0
                                        and pt1[1] > 0
                                        and pt2[0] > 0
                                        and pt2[1] > 0
                                    ):
                                        cv2.line(
                                            display_frame, pt1, pt2, (0, 255, 0), 2
                                        )
                            # Draw keypoints
                            for x, y in kps:
                                if x > 0 and y > 0:  # Only draw valid keypoints
                                    cv2.circle(
                                        display_frame,
                                        (int(x), int(y)),
                                        3,
                                        (0, 255, 0),
                                        -1,
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
                            f"Person: {status} | Faces: {current_detection.get('face_count', 0)} | Motion blobs: {current_detection.get('motion_count', 0)}",
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
            if video_writer:
                video_writer.release()
            if show_frames:
                cv2.destroyAllWindows()

            # Determine start and end times using person detections
            # Start: First person detected walking / moving (any person indicator)
            # End: Last person detected (last face that disappears)

            # 1) Find periods where people are walking (sustained person detections)
            walking_periods = []  # List of (start_time, end_time) for each person walking
            current_walking_start = None
            # Use a shorter window so we also catch people who are already close to camera
            # when the video starts, or who only appear briefly at the beginning.
            min_walking_duration = 0.25
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

            # 2) Start time: detect when someone STARTS WALKING/MOVING
            #    Prefer per-person tracking (ID-based) when available, then fall back
            #    to the older global motion logic.
            segment_start_time = None

            # 2a) Per-person walking starts from YOLO tracking (highest priority)
            walking_starts: List[float] = [
                state["walking_start_time"]
                for state in person_states.values()
                if state.get("walking_start_time") is not None
            ]

            if walking_starts:
                # Earliest person who went from stationary → walking
                segment_start_time = min(walking_starts)
                print(
                    f"[kitting] using per-person walking start: {segment_start_time:.2f}s "
                    f"(from {len(walking_starts)} people)"
                )

            # Establish baseline: what's the typical motion in the first few seconds?
            # This helps us filter out people who are already standing still
            baseline_window = min(
                3.0, video_duration * 0.15
            )  # First 3s or 15% of video
            baseline_motion_areas = []
            baseline_has_any_person = False

            for detection in person_detections:
                if detection["time"] <= baseline_window:
                    if detection["has_motion"]:
                        baseline_motion_areas.append(detection["motion_area"])
                    if detection["has_person_body"] or detection["has_face"]:
                        baseline_has_any_person = True
                else:
                    break

            # Calculate baseline motion threshold (median of early motion)
            baseline_motion_threshold = 0.0
            if baseline_motion_areas:
                baseline_motion_areas.sort()
                baseline_motion_threshold = baseline_motion_areas[
                    len(baseline_motion_areas) // 2
                ]

            # Now find the first time someone STARTS moving (not just standing)
            # We look for:
            # 1. Motion that's significantly larger than baseline (new person moving)
            # 2. Motion that appears after a period of no/low motion
            # 3. YOLO body detected WITH motion (person moving, not just standing)
            motion_appearance_window = 1.5  # Look back 1.5s to check if motion is "new"

            for i, detection in enumerate(person_detections):
                # Skip the baseline window - we're establishing what's "normal"
                if detection["time"] <= baseline_window:
                    continue

                is_starting_to_move = False

                # Method 1: YOLO body + motion = person actively moving (not just standing)
                if detection["has_person_body"] and detection["has_motion"]:
                    is_starting_to_move = True

                # Method 2: Significant motion that's NEW (larger than baseline)
                elif detection["has_motion"]:
                    # Check if this motion is significantly larger than baseline
                    if (
                        not baseline_has_any_person
                        or detection["motion_area"] > baseline_motion_threshold * 2.0
                    ):
                        # Also check if this is "new motion" - wasn't there before
                        window_start_time = detection["time"] - motion_appearance_window
                        had_similar_motion_before = False

                        for prev_detection in person_detections:
                            if prev_detection["time"] < window_start_time:
                                continue
                            if prev_detection["time"] >= detection["time"]:
                                break
                            # Check if there was similar or larger motion before
                            if (
                                prev_detection["has_motion"]
                                and prev_detection["motion_area"]
                                >= detection["motion_area"] * 0.8
                            ):
                                had_similar_motion_before = True
                                break

                        # If no similar motion before, this is NEW motion (someone starting)
                        if not had_similar_motion_before:
                            is_starting_to_move = True

                # Method 3: Motion that grows significantly (someone starting to walk)
                elif detection["has_motion"] and i > 0:
                    # Compare with previous detection to see if motion is growing
                    prev_detection = person_detections[i - 1]
                    if prev_detection["has_motion"]:
                        growth_factor = detection["motion_area"] / max(
                            prev_detection["motion_area"], 1.0
                        )
                        if growth_factor >= 1.5:  # Motion grew by 50%+
                            is_starting_to_move = True

                if is_starting_to_move:
                    segment_start_time = detection["time"]
                    break

            # Fallback: if no "starting motion" detected, use first sustained walking period
            if segment_start_time is None and walking_periods:
                segment_start_time = walking_periods[0][0]

            # Final fallback - first face with motion, or first face, or 0.0
            if segment_start_time is None:
                for detection in person_detections:
                    if detection["has_face"] and detection["has_motion"]:
                        segment_start_time = detection["time"]
                        break
                else:
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
                    capture_images_min_delay,
                    platform_number,
                    direction,
                    show_progress,
                    show_frames,
                    capture_offset_after_start,
                    capture_offset_before_end,
                    capture_images_output_dir,
                    filename_prefix,
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
                    capture_images_min_delay,
                    platform_number,
                    direction,
                    show_progress,
                    show_frames,
                    capture_offset_after_start,
                    capture_offset_before_end,
                    capture_images_output_dir,
                    filename_prefix,
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
                    capture_images_min_delay,
                    platform_number,
                    direction,
                    show_progress,
                    show_frames,
                    capture_offset_after_start,
                    capture_offset_before_end,
                    capture_images_output_dir,
                    filename_prefix,
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
        input_video_path="new-videos/GX010469.MP4",
        platform_number=3,
        show_frames=True,
    )

    import json

    print(json.dumps(result, indent=2))
