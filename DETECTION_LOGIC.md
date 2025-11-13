# Face Detection Logic Documentation

This document explains the detection logic, start/end time determination, and duration rules for both "coming" and "going" video directions.

---

## "Coming" Direction Videos

### Overview

For "coming" videos, the rider comes from a higher platform down to a lower platform where the camera is positioned. The system detects when the rider enters the frame and tracks until their motion stops.

### Detection Methods

#### 1. Motion Detection (Background Subtraction)

- Uses OpenCV's MOG2 background subtractor to detect moving objects
- Applies morphological operations to reduce noise
- Filters out the stationary guide by analyzing motion patterns

#### 2. Face Detection (MediaPipe)

- Uses MediaPipe Face Detection model (full-range, model_selection=1)
- Minimum detection confidence: 0.5
- Minimum face width: 6% of frame width
- When multiple faces are detected, selects the one with the largest bounding box

### Start Time Logic

The start time is determined by detecting the **rider's motion** (not the guide's):

1. **Collect all motion samples** throughout the video
2. **Filter out the guide's motion**:
   - Guide motion: Small and constant (stationary)
   - Rider motion: Grows over time as they approach
3. **Detection strategy**:
   - Looks for motion that **grows by at least 2x** over a short window (~0.5 seconds)
   - This indicates the rider approaching, not the stationary guide
4. **Fallback**:
   - If no growing motion is found, uses a higher threshold (15% of max area) to filter out guide's small constant motion
   - If still no rider motion detected: Uses `end_time - min_duration` as start time

**Result**: Start time = First detection of rider motion (filtered from guide)

### End Time Logic

The end time is determined by **motion detection** (not face detection) to avoid capturing guide activity:

1. **Track motion after start time**:
   - Looks for when motion becomes very small (< 5% of max detected area)
   - Allows brief gaps (~0.3 seconds) to handle brief occlusions
2. **Motion stop detection**:
   - When motion stops or becomes very small for 3 consecutive samples, that becomes the base end time
3. **Smoother finish extension**:
   - Extends by **1.5 seconds** after motion stops
   - Provides smoother finish without capturing guide activity (belt removal, etc.)
   - Caps at video duration

**Result**: End time = When rider motion stops + 1.5 seconds extension

### Duration Rules

1. **If duration > max_duration**:

   - Trims from the **end** to reach `max_duration`
   - Formula: `end_time = start_time + max_duration`

2. **If duration < min_duration**:

   - Tries to extend **forward** (later in video) if possible
   - If enough video remains: Extends end time to meet `min_duration`
   - If not enough video: Extends as much as possible (to video end)
   - If still too short after maximum extension: Returns invalid result

3. **Final validation**:
   - Ensures final duration is within `min_duration` and `max_duration` bounds
   - If valid, returns the segment with rounded times

---

## "Going" Direction Videos

### Overview

For "going" videos, the rider is at the lower platform and looks at the camera before going up. The system detects when the rider first looks at the camera and tracks until the end of the video.

### Detection Methods

#### Face Detection (MediaPipe)

- Uses MediaPipe Face Detection model (full-range, model_selection=1)
- Minimum detection confidence: 0.5
- Minimum face width: 6% of frame width
- When multiple faces are detected, selects the one with the **largest bounding box**
- Requires **3 consecutive stable detections** (at least 0.5 seconds apart) to record a detection

### Start Time Logic

The start time is determined by finding the **optimal face detection** that creates a clip closest to the desired duration:

1. **Collect all face detections** throughout the video
2. **Filter stable detections**:
   - Only records detections that are stable (3 consecutive hits)
   - Minimum 0.5 seconds between recorded detections
3. **Selection strategy**:
   - Evaluates each face detection as a potential start time
   - Calculates what the duration would be if starting at that detection
   - Selects the detection that makes the clip duration **closest to `min_duration`**
   - Prefers durations that are >= `min_duration` and <= `max_duration` (if set)
4. **Fallback**:
   - If no face detected: Uses `end_time - min_duration` as start time
   - If no detection gives valid duration: Uses the earliest detection

**Result**: Start time = Face detection that produces clip closest to `min_duration`

### End Time Logic

The end time is **always the end of the video**:

- `end_time = video_duration`
- This captures the full segment from when the rider looks at the camera until the video ends

### Duration Rules

1. **If duration > max_duration**:

   - Trims from the **end** to reach `max_duration`
   - Formula: `end_time = start_time + max_duration`

2. **If duration < min_duration**:

   - Tries to extend **forward** (later in video) if possible
   - If enough video remains: Extends end time to meet `min_duration`
   - If not enough video: Extends as much as possible (to video end)
   - If still too short after maximum extension: Returns invalid result

3. **Final validation**:
   - Ensures final duration is within `min_duration` and `max_duration` bounds
   - If valid, returns the segment with rounded times

---

## Common Duration Rules (Both Directions)

### Duration Constraints Application Order

1. **Initial calculation**: `duration = end_time - start_time`

2. **Apply max_duration constraint** (if set):

   - If `duration > max_duration`: Trim from end
   - `end_time = start_time + max_duration`

3. **Apply min_duration constraint**:

   - If `duration < min_duration`: Try to extend
   - Extension direction:
     - **Coming**: Extend forward (later in video)
     - **Going**: Extend forward (later in video)
   - If extension not possible: Use maximum available duration

4. **Final validation**:
   - Check if final duration meets `min_duration` requirement
   - If not: Return invalid result with reason

### Return Values

**Valid Result**:

```json
{
  "input_video": "path/to/video.mp4",
  "direction": "coming" | "going",
  "start_time": 2.5,
  "end_time": 12.3,
  "duration": 9.8,
  "valid": true,
  "output_video": "path/to/output.mp4"  // if provided
}
```

**Invalid Result**:

```json
{
  "input_video": "path/to/video.mp4",
  "direction": "coming" | "going",
  "valid": false,
  "reason": "Detected duration 1.5s is below minimum 2.0s"
}
```

---

## Technical Details

### Frame Sampling

- Samples frames at ~10 frames per second (every 0.1 seconds) for efficiency
- Formula: `sample_interval = max(1, int(fps * 0.1))`

### Motion Detection Parameters

- Background subtractor: MOG2 with history=500, varThreshold=50
- Minimum motion area: 1% of frame area
- Growth detection window: 5 samples (~0.5 seconds)
- Minimum growth factor: 2.0x (area must double to be considered rider)
- Motion stop threshold: 5% of maximum detected area
- Gap tolerance: 3 samples (~0.3 seconds)

### Face Detection Parameters

- Model: MediaPipe Face Detection (full-range)
- Confidence threshold: 0.5
- Minimum face size: 6% of frame width
- Stability requirement: 3 consecutive detections
- Minimum time between detections: 0.5 seconds

### Extension Times

- **Coming**: 1.5 seconds after motion stops
- **Going**: N/A (uses video end)

---

## Summary

| Aspect              | Coming                                | Going                             |
| ------------------- | ------------------------------------- | --------------------------------- |
| **Start Detection** | Rider motion (growing)                | Face detection (optimal)          |
| **End Detection**   | Motion stops + 1.5s                   | Video end                         |
| **Primary Method**  | Motion analysis                       | Face detection                    |
| **Filtering**       | Filters out guide's stationary motion | Filters out small/weak detections |
| **Extension**       | 1.5s after motion stops               | N/A                               |
| **Duration Target** | Between min and max                   | Closest to min                    |
