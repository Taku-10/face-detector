"""
Zipline Video Time Detection

Automatically detects when a zipline rider appears and exits the frame,
returning the start and end timestamps for that segment.
"""

import cv2
from typing import Optional


def detect_zipline_segment(
    input_video_path: str,
    direction: str = "coming",
    min_duration: float = 2.0,
    max_duration: Optional[float] = None,
) -> dict:
    """
    Detects the visible time range of a zipline rider in the video.

    Args:
        input_video_path: Path to the raw video file
        direction: Either "coming" or "going" (direction of rider)
        min_duration: Minimum motion duration to be considered valid (seconds)
        max_duration: Optional maximum time cap (seconds)

    Returns:
        dict with:
            - input_video: path to input video
            - direction: detected/requested direction
            - start_time: when subject first enters frame (seconds)
            - end_time: when subject leaves or movement ends (seconds)
            - duration: total duration (seconds)
            - valid: bool indicating if detection meets criteria
            - reason: optional reason if invalid
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

        if direction == "coming":
            # For "coming": detect rider approaching (motion that grows over time)
            # Strategy: Process forward, look for motion that starts small and grows
            # This filters out guide's constant small motion
            
            motion_samples = []
            frame_count = 0
            
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
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)
                        
                        # Threshold for significant movement
                        min_area = (frame.shape[0] * frame.shape[1]) * 0.01  # 1% of frame
                        
                        if area > min_area:
                            has_motion = True
                            x, y, w, h = cv2.boundingRect(largest_contour)
                            motion_area = float(w * h)
                    
                    motion_samples.append({
                        "time": frame_time,
                        "has_motion": has_motion,
                        "area": motion_area,
                    })
                
                frame_count += 1
            
            cap.release()
            
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
                future_samples = motion_samples[i:i+growth_window_size]
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
                (i for i, s in enumerate(motion_samples) if s["time"] >= segment_start_time),
                len(motion_samples) - 1
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
                
                return {
                    "input_video": input_video_path,
                    "direction": direction,
                    "start_time": round(segment_start_time, 2),
                    "end_time": round(segment_end_time, 2),
                    "duration": round(duration, 2),
                    "valid": True,
                }
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
            
            # Load face detection cascades
            frontal_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )
            
            if frontal_cascade.empty():
                cap.release()
                return {
                    "input_video": input_video_path,
                    "direction": direction,
                    "valid": False,
                    "reason": "Failed to load frontal face cascade",
                }
            
            # Face detection parameters
            min_face_width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.06)  # 6% of frame width
            min_consecutive_hits = 3  # Need 3 consecutive detections for stability
            consecutive_hits = 0
            segment_start_time = None
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_time = frame_count / fps
                
                # Sample frames for efficiency
                if frame_count % sample_interval == 0:
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.equalizeHist(gray, gray)
                    
                    # Detect frontal faces
                    faces = frontal_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(min_face_width_px, min_face_width_px),
                        flags=cv2.CASCADE_SCALE_IMAGE,
                    )
                    
                    # Check if we have a stable frontal face (with eyes visible)
                    detected_stable_frontal = False
                    for (x, y, w, h) in faces:
                        # Check for eyes to confirm it's a good frontal face
                        face_roi_gray = gray[y : y + h, x : x + w]
                        eyes = eye_cascade.detectMultiScale(
                            face_roi_gray,
                            scaleFactor=1.1,
                            minNeighbors=3,
                            flags=cv2.CASCADE_SCALE_IMAGE,
                            minSize=(int(w * 0.15), int(h * 0.15)),
                        )
                        # At least one eye visible indicates good frontal face
                        if len(eyes) >= 1:
                            detected_stable_frontal = True
                            break
                    
                    if detected_stable_frontal:
                        consecutive_hits += 1
                        # Lock start time when we have stable detection
                        if consecutive_hits >= min_consecutive_hits and segment_start_time is None:
                            segment_start_time = frame_time
                            # Don't break - we need to continue to get video end time
                    else:
                        # Reset counter if we haven't locked start yet
                        if segment_start_time is None:
                            consecutive_hits = 0
                
                frame_count += 1
            
            cap.release()
            
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
                
                return {
                    "input_video": input_video_path,
                    "direction": direction,
                    "start_time": round(segment_start_time, 2),
                    "end_time": round(segment_end_time, 2),
                    "duration": round(duration, 2),
                    "valid": True,
                }
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
        input_video_path="GH011700.MP4",  # Change this to your video path
        direction="going",  # or "going"
        min_duration=2.0,
        max_duration=20.0,  # Optional: set to None for no limit
    )

    # Print results
    import json

    print(json.dumps(result, indent=2))
