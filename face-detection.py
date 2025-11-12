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
    min_duration: float = 2.0,
    max_duration: Optional[float] = None,
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
            # For "coming": detect rider approaching (motion that grows over time)
            # Strategy: Process forward, look for motion that starts small and grows
            # This filters out guide's constant small motion

            motion_samples = []
            frame_count = 0

            # Track last detection state for smooth video overlay
            last_contours = []
            last_motion_box = None
            last_motion_area = 0

            # First pass: collect all motion data
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_time = frame_count / fps

                # Sample frames
                if frame_count % sample_interval == 0:
                    # Apply background subtraction
                    fg_mask = bg_subtractor.apply(frame)

                    # Morphological operations to reduce noise
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                    # Find contours
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

                    motion_samples.append(
                        {
                            "time": frame_time,
                            "has_motion": has_motion,
                            "area": motion_area,
                        }
                    )

                # Create display frame with overlays if needed
                if show_frames or output_video_path:
                    display_frame = frame.copy()

                    # Draw all contours from last detection
                    if last_contours:
                        cv2.drawContours(
                            display_frame, last_contours, -1, (0, 255, 0), 2
                        )

                    # Draw bounding box from last significant motion
                    if last_motion_box:
                        x, y, w, h = last_motion_box
                        cv2.rectangle(
                            display_frame, (x, y), (x + w, y + h), (0, 0, 255), 3
                        )
                        cv2.putText(
                            display_frame,
                            f"Area: {int(last_motion_area)}",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
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

                # Show frame with overlay (with longer delay to reduce lag)
                if show_frames:
                    cv2.imshow("Motion Detection", display_frame)
                    # Increased waitKey to reduce lag - waits 30ms instead of 1ms
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        cap.release()
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
            if video_writer:
                video_writer.release()
            if show_frames:
                cv2.destroyAllWindows()

            # Find motion samples with significant area
            motion_with_area = [s for s in motion_samples if s["area"] > 0]
            if not motion_with_area:
                return {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": "No motion detected in video",
                }

            # Find max area to determine thresholds
            max_area = max(s["area"] for s in motion_with_area)

            # For "coming", rider motion should GROW over time (small to large)
            # Guide motion stays small and constant
            # Strategy: Look for motion that shows growth pattern

            segment_start_time = None
            segment_end_time = None

            # Look for motion that grows significantly (rider approaching)
            # Track area progression forward in time
            growth_window_size = 5  # Check growth over 5 samples (~0.5 seconds)
            min_growth_factor = 2.0  # Area must grow by at least 2x to be rider

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
                guide_filter_threshold = max_area * 0.15  # Must be at least 15% of max

                for sample in motion_samples:
                    if sample["area"] >= guide_filter_threshold:
                        segment_start_time = sample["time"]
                        break

            if segment_start_time is None:
                return {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": "Could not detect rider motion (filtered out guide motion)",
                }

            # Find end time: when motion stops or becomes very small
            # Work forward from start to find when motion ends
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
                        segment_end_time = motion_samples[last_motion_idx]["time"]
                        break

            # If motion continues to end of video
            if segment_end_time is None:
                segment_end_time = motion_samples[last_motion_idx]["time"]

            # Calculate duration
            duration = segment_end_time - segment_start_time

            if duration >= min_duration:
                # Apply max_duration cap if specified
                if max_duration is not None and duration > max_duration:
                    segment_start_time = segment_end_time - max_duration
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
            # For "going": detect when rider first looks at camera (frontal face)
            # Start time = when stable frontal face is detected
            # End time = end of video

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0.0

            # Initialize MediaPipe Face Detection
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5
            )

            # Face detection parameters
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            min_face_width_px = int(frame_width * 0.06)  # 6% of frame width
            min_consecutive_hits = 3  # 3 consecutive detections for stability
            consecutive_hits = 0
            segment_start_time = None

            # Track last detection state for smooth video overlay
            last_faces = []
            last_detection_confidence = 0.0
            last_detected_stable = False

            frame_count = 0

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

                    # Store detection results and check for stable frontal face
                    detected_stable_frontal = False
                    last_faces = []
                    last_detection_confidence = 0.0

                    if results.detections:
                        # Select the detection with largest bounding box as primary
                        # This helps when multiple faces are detected (e.g., head shaking, multiple angles)
                        primary_detection = max(
                            results.detections,
                            key=lambda det: (
                                det.location_data.relative_bounding_box.width *
                                det.location_data.relative_bounding_box.height
                            )
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
                            # MediaPipe face detection is already quite good at detecting frontal faces
                            # We use confidence threshold to ensure quality detection
                            if confidence >= 0.5:  # MediaPipe confidence threshold
                                detected_stable_frontal = True
                                last_faces.append((x, y, w, h))
                                last_detection_confidence = confidence

                    last_detected_stable = detected_stable_frontal

                    if detected_stable_frontal:
                        consecutive_hits += 1
                        # Lock start time when we have stable detection
                        if (
                            consecutive_hits >= min_consecutive_hits
                            and segment_start_time is None
                        ):
                            segment_start_time = frame_time
                            # Don't break - we need to continue to get video end time
                    else:
                        # Reset counter if we haven't locked start yet
                        if segment_start_time is None:
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
                        f"Consecutive: {consecutive_hits}/{min_consecutive_hits}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    if segment_start_time is not None:
                        cv2.putText(
                            display_frame,
                            f"Start Locked: {segment_start_time:.2f}s",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                # Show frame with overlay (with longer delay to reduce lag)
                if show_frames:
                    cv2.imshow("Face Detection", display_frame)
                    # Increased waitKey to reduce lag - waits 30ms instead of 1ms
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        cap.release()
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

            # End time is the end of the video
            segment_end_time = video_duration

            if segment_start_time is None:
                return {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": "No stable frontal face detected (rider did not look at camera)",
                }

            # Calculate duration
            duration = segment_end_time - segment_start_time

            if duration >= min_duration:
                # Apply max_duration cap if specified
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
        input_video_path="fg-2.MP4",  # Change this to your video path
        direction="going",  # or "coming"
        min_duration=2.0,
        max_duration=20.0,  # Optional: set to None for no limit
        show_frames=True,  # Set to True to display detection in real-time
        # output_video_path="output_with_detections.mp4",  # Optional: save video with overlays
    )

    # Print results
    import json

    print(json.dumps(result, indent=2))
