"""
Zipline Video Time Detection

Automatically detects when a zipline rider appears and exits the frame,
returning the start and end timestamps for that segment.
"""

import cv2
import mediapipe as mp
from typing import Optional


def detect_zipline_segment(
    input_video_path: str,
    direction: str = "coming",
    min_duration: float = 4.0,
    max_duration: float = 10.0,
    ideal_duration: float = 8.0,
    show_frames: bool = False,
    output_video_path: Optional[str] = None,
) -> dict:
    """
    Detects the visible time range of a zipline rider in the video.

    Args:
        input_video_path: Path to the raw video file
        direction: Either "coming" or "going" (direction of rider)
        min_duration: Minimum motion duration to be considered valid (seconds)
        max_duration: Optional maximum time cap (seconds)
        ideal_duration: Ideal duration for the clip (default: 8.0 seconds)
            - For "going" videos: system picks face detection that gets closest to this duration
        show_frames: If True, displays frames with detection overlay in real-time (default: False)
        output_video_path: Optional path to save video with detection overlay (default: None)

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

    Detection Logic:

    "COMING" Videos:
    - Detects when rider approaches from higher platform to camera position
    - Start: First significant motion detection (rider entering frame, filtered from guide motion)
    - End: When rider motion stops + 1.5s extension for smoother finish
    - Motion detection uses background subtraction to find growing motion patterns
    - Filters out guide's constant small motion, focuses on rider's growing motion
    - Duration constraints: Must be between min_duration and max_duration

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
    """
    # Validate inputs
    if direction not in ["coming", "going"]:
        return {
            "input_video": input_video_path,
            "direction": direction,
            "valid": False,
            "reason": f"Invalid direction: {direction}. Must be 'coming' or 'going'",
        }

    try:
        # Open video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            return {
                "input_video": input_video_path,
                "direction": direction,
                "valid": False,
                "reason": "Could not open video file",
            }

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return {
                "input_video": input_video_path,
                "direction": direction,
                "valid": False,
                "reason": "Invalid FPS in video file",
            }

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
                        return {
                            "input_video": input_video_path,
                            "direction": direction,
                            "valid": False,
                            "reason": "Detection stopped by user",
                        }

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

                segment_end_time = None  # Will be determined below

                if segment_start_time is None:
                    # No rider motion detected - fallback: use last min_duration seconds
                    segment_start_time = max(0.0, video_duration - min_duration)
                    segment_end_time = video_duration
                else:
                    # Start = first rider detection (filtered from guide)
                    # End = when rider motion stops (more reliable than face detection)
                    # Find when motion stops after the start time
                    start_idx = next(
                        (
                            i
                            for i, s in enumerate(motion_samples)
                            if s["time"] >= segment_start_time
                        ),
                        len(motion_samples) - 1,
                    )

                    # Look for when motion stops or becomes very small
                    # Allow for brief gaps (rider might be briefly occluded)
                    max_area = max(s["area"] for s in motion_with_area)
                    min_end_area = max_area * 0.05  # 5% of max - very small
                    gap_tolerance = 3  # Allow up to 3 samples (~0.3s) of no motion

                    last_motion_idx = start_idx
                    gap_count = 0

                    for i in range(start_idx, len(motion_samples)):
                        sample = motion_samples[i]

                        if sample["area"] >= min_end_area:
                            # Motion detected
                            last_motion_idx = i
                            gap_count = 0
                        else:
                            gap_count += 1
                            if gap_count >= gap_tolerance:
                                # Motion has stopped
                                segment_end_time = motion_samples[last_motion_idx][
                                    "time"
                                ]
                                break

                    # If motion continues to end of video
                    if segment_end_time is None:
                        segment_end_time = (
                            motion_samples[last_motion_idx]["time"]
                            if last_motion_idx < len(motion_samples)
                            else video_duration
                        )

                    # Ensure segment_end_time is set
                    if segment_end_time is None:
                        segment_end_time = video_duration

                    # Extend by 1-2 seconds after motion stops for smoother finish
                    extension_time = 1.5  # 1.5 seconds extension
                    segment_end_time = min(
                        video_duration, segment_end_time + extension_time
                    )

            # Calculate initial duration
            duration = segment_end_time - segment_start_time

            # Apply duration constraints
            if max_duration is not None and duration > max_duration:
                # Trim from the end to reach max_duration
                segment_end_time = segment_start_time + max_duration
                duration = max_duration

            if duration < min_duration:
                # Try to extend forward (later in video) if possible
                available_forward = video_duration - segment_end_time
                needed_extension = min_duration - duration

                if available_forward >= needed_extension:
                    # Can extend forward
                    segment_end_time = segment_end_time + needed_extension
                    duration = min_duration
                else:
                    # Can't extend enough, extend as much as possible
                    segment_end_time = video_duration
                    duration = segment_end_time - segment_start_time

                    # If still too short, we'll return invalid
                    if duration < min_duration:
                        return {
                            "input_video": input_video_path,
                            "direction": direction,
                            "valid": False,
                            "reason": f"Detected duration {duration:.2f}s is below minimum {min_duration}s (even after extending)",
                        }

            # Final validation
            if duration >= min_duration:
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
                if output_video_path:
                    result["output_video"] = output_video_path
                return result
            else:
                return {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": f"Detected duration {duration:.2f}s is below minimum {min_duration}s",
                }

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
                        return {
                            "input_video": input_video_path,
                            "direction": direction,
                            "valid": False,
                            "reason": "Detection stopped by user",
                        }

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

            # Calculate initial duration
            duration = segment_end_time - segment_start_time

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
                # Rule 2: Try to extend forward (backward in time) if possible
                available_backward = segment_start_time  # How much time before start
                needed_extension = min_duration - duration

                if available_backward >= needed_extension:
                    # Can extend backward enough
                    segment_start_time = segment_start_time - needed_extension
                    duration = min_duration
                else:
                    # Can't extend enough backward, extend as much as possible
                    segment_start_time = 0.0
                    duration = segment_end_time - segment_start_time

                    # If still too short after extending, check if we can use full segment
                    if duration < min_duration:
                        # Use full segment if it's at least somewhat reasonable
                        if (
                            duration >= min_duration * 0.5
                        ):  # At least 50% of min_duration
                            # Accept the shorter segment
                            pass
                        else:
                            return {
                                "input_video": input_video_path,
                                "direction": direction,
                                "valid": False,
                                "reason": f"Detected duration {duration:.2f}s is below minimum {min_duration}s (even after extending)",
                            }

            # Final validation: ensure we respect min and max constraints
            if duration >= min_duration:
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
                if output_video_path:
                    result["output_video"] = output_video_path
                return result
            else:
                return {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": f"Detected duration {duration:.2f}s is below minimum {min_duration}s",
                }

    except Exception as e:
        return {
            "input_video": input_video_path,
            "direction": direction,
            "valid": False,
            "reason": f"Error processing video: {str(e)}",
        }


if __name__ == "__main__":
    # Example usage - modify these values to test with your video
    result = detect_zipline_segment(
        input_video_path="coming-new-3.MP4",  # Change this to your video path
        direction="going",  # or "coming"
        min_duration=2.0,
        max_duration=20.0,
        ideal_duration=8.0,  # Ideal duration for "going" videos (default: 8.0 seconds)
        show_frames=True,  # Set to True to display detection in real-time
        # output_video_path="output_with_detections.mp4",  # Optional: save video with overlays
    )

    # Print results
    import json

    print(json.dumps(result, indent=2))
