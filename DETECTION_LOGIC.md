# Face Detection Logic Documentation

This document explains the detection logic, start/end time determination, and duration rules for both "coming" and "going" video directions.

---

## Parameters

### Core Parameters

- **`min_duration`**: Minimum duration for a valid clip (default: 2.0 seconds)
- **`max_duration`**: Maximum duration cap (optional, can be `None`)
- **`ideal_duration`**: Ideal duration for clips (default: 8.0 seconds)
  - Used primarily for "going" videos to select the best face detection
  - Ensures clips are close to the desired length when possible

---

## "COMING" Direction Videos

### Overview

For "coming" videos, the rider approaches from a higher platform down to the lower platform where the camera is positioned. The system detects the rider's motion and face to determine when they enter the frame.

### Detection Methods

#### 1. Motion Detection (Background Subtraction)

- Uses OpenCV's MOG2 background subtractor to detect moving objects
- Applies morphological operations (close and open) to reduce noise
- Samples frames at ~10 frames per second for efficiency
- Filters out the stationary guide by analyzing motion growth patterns

#### 2. Face Detection (MediaPipe)

- Uses MediaPipe Face Detection model (full-range, `model_selection=1`)
- Minimum detection confidence: 0.5
- Minimum face width: 6% of frame width
- When multiple faces are detected, selects the one with the largest bounding box
- Records face detection times for correlation with motion patterns

### Start Time Logic

The start time is determined by detecting **rider motion** (not guide motion):

1. **Collect motion samples** throughout the video
   - Each sample contains: time, has_motion flag, motion area
   - Motion area is calculated from the largest contour

2. **Identify rider's motion** (filters out guide):
   - **Primary method**: Look for motion that **grows significantly** over time
     - Growth window: 5 samples (~0.5 seconds)
     - Growth factor: Motion area must grow by at least 2x
     - This identifies the rider approaching (motion grows) vs. guide (constant small motion)
   - **Fallback method**: If no growing motion found, use first significant motion
     - Threshold: Motion must be at least 15% of maximum detected area
     - This filters out guide's constant small motion

3. **Set start time**:
   - Start time = Time of first detected rider motion
   - If no rider motion detected: Fallback to `video_duration - min_duration`

**Result**: Start time = First significant rider motion detection

### End Time Logic

The end time is determined by when **rider motion stops**:

1. **Find motion stop point**:
   - Starting from the detected start time, look for when motion stops
   - Motion stop threshold: 5% of maximum detected area
   - Gap tolerance: Allows up to 3 samples (~0.3s) of no motion before considering motion stopped

2. **Add extension**:
   - Extends end time by **1.5 seconds** after motion stops
   - This provides a smoother finish to the clip
   - Clamped to video duration: `end_time = min(video_duration, motion_stop_time + 1.5)`

3. **Fallback**:
   - If motion continues to end of video: `end_time = video_duration`
   - If no motion detected: `end_time = video_duration`

**Result**: End time = Motion stop time + 1.5 seconds (or video end)

### Duration Rules for "Coming" Videos

After calculating initial start and end times, the following rules are applied:

#### Rule 1: Apply max_duration constraint (if set)

- **If `duration > max_duration`**:
  - Trim from the **end** to reach `max_duration`
  - Formula: `end_time = start_time + max_duration`
  - `duration = max_duration`

#### Rule 2: Apply min_duration constraint

- **If `duration < min_duration`**:
  - Try to extend **forward** (later in video) if possible
  - Calculate available forward time: `available_forward = video_duration - end_time`
  - Calculate needed extension: `needed_extension = min_duration - duration`
  - **If `available_forward >= needed_extension`**:
    - Extend end time forward: `end_time = end_time + needed_extension`
    - `duration = min_duration`
  - **If `available_forward < needed_extension`**:
    - Extend as much as possible: `end_time = video_duration`
    - `duration = end_time - start_time`
    - If still too short: Return invalid result

#### Rule 3: Final validation

- Ensures final duration is within `min_duration` and `max_duration` bounds
- If valid, returns the segment with rounded times (2 decimal places)

**Note**: `ideal_duration` is **not used** for "coming" videos. Only `min_duration` and `max_duration` constraints apply.

---

## "GOING" Direction Videos

### Overview

For "going" videos, the rider is at the lower platform and looks at the camera before going up. The system detects when the rider looks at the camera and selects the optimal detection to create a clip closest to the ideal duration.

### Detection Methods

#### Face Detection (MediaPipe)

- Uses MediaPipe Face Detection model (full-range, `model_selection=1`)
- Minimum detection confidence: 0.5
- Minimum face width: 6% of frame width
- When multiple faces are detected, selects the one with the **largest bounding box**
- Requires **3 consecutive stable detections** to record a detection
- Minimum 0.5 seconds between recorded detections (prevents duplicate detections)

### Start Time Logic

The start time is determined by finding the **optimal face detection** that creates a clip closest to `ideal_duration`:

1. **Collect all face detections** throughout the video
   - Each detection is recorded with: time, confidence, bounding box
   - Only stable detections (3 consecutive hits) are recorded
   - Detections must be at least 0.5 seconds apart

2. **Select optimal detection**:
   - **If face detections found**:
     - For each face detection, calculate what the duration would be if starting at that time
     - Formula: `candidate_duration = video_duration - detection_time`
     - Calculate difference from ideal: `duration_diff = abs(candidate_duration - ideal_duration)`
     - Select the detection with the **smallest duration difference** (closest to ideal)
     - This ensures the clip is as close as possible to `ideal_duration`
   - **If no face detected**:
     - Use segment from **0 to `ideal_duration`** (clamped to video length)
     - `segment_start_time = 0.0`
     - `segment_end_time = min(video_duration, ideal_duration)`

3. **Fallback**:
   - If no detection gives valid result: Use the earliest detection

**Result**: Start time = Face detection that produces clip closest to `ideal_duration`

### End Time Logic

The end time depends on whether faces were detected:

- **If faces detected**: `end_time = video_duration` (always end of video)
- **If no faces detected**: `end_time = min(video_duration, ideal_duration)`

**Result**: End time = End of video (when faces detected) or `ideal_duration` (when no faces)

### Duration Rules for "Going" Videos

After calculating initial start and end times, the following rules are applied **in order**:

#### Rule 1: Apply max_duration constraint (if set)

- **If `duration > max_duration`**:
  - **If `ideal_duration <= max_duration`**:
    - Trim from the **end** to reach `ideal_duration`
    - Formula: `end_time = start_time + ideal_duration`
    - `duration = ideal_duration`
  - **If `ideal_duration > max_duration`**:
    - Ideal exceeds max, so trim to `max_duration` instead
    - Formula: `end_time = start_time + max_duration`
    - `duration = max_duration`

#### Rule 2: Apply min_duration constraint

- **If `duration < min_duration`**:
  - Try to extend **backward** (earlier in video) if possible
  - Calculate available backward time: `available_backward = start_time`
  - Calculate needed extension: `needed_extension = min_duration - duration`
  - **If `available_backward >= needed_extension`**:
    - Extend start time backward: `start_time = start_time - needed_extension`
    - `duration = min_duration`
  - **If `available_backward < needed_extension`**:
    - Extend as much as possible: `start_time = 0.0`
    - `duration = end_time - start_time`
    - **If still too short**:
      - If `duration >= min_duration * 0.5` (at least 50% of min): Accept the shorter segment
      - Otherwise: Return invalid result

#### Rule 3: Final validation

- Ensures final duration is within `min_duration` and `max_duration` bounds
- If `max_duration` is set and duration still exceeds it: Final trim to `max_duration`
- If valid, returns the segment with rounded times (2 decimal places)

**Key Points**:
- `ideal_duration` is the **primary target** for "going" videos
- System selects face detection that gets closest to `ideal_duration`
- If duration exceeds `max_duration`, trims to `ideal_duration` (if within max) or `max_duration`
- If duration is below `min_duration`, extends backward in time

---

## Comparison Table

| Aspect | Coming | Going |
|--------|--------|-------|
| **Start Detection** | Rider motion (growing pattern) | Face detection (optimal for ideal_duration) |
| **End Detection** | Motion stop + 1.5s extension | Video end (or ideal_duration if no faces) |
| **Primary Method** | Motion + face correlation | Face detection only |
| **Filtering** | Filters guide's constant motion | Filters small/weak detections |
| **Extension** | 1.5s after motion stops | N/A |
| **Duration Target** | Between min and max | Closest to ideal_duration |
| **Uses ideal_duration** | ❌ No | ✅ Yes |
| **No Detection Fallback** | `end_time - min_duration` | `0 to ideal_duration` |

---

## Duration Rules Summary

### "Coming" Videos

1. **Initial**: `duration = end_time - start_time`
2. **If `duration > max_duration`**: Trim from end → `duration = max_duration`
3. **If `duration < min_duration`**: Extend forward (later) → `duration = min_duration` (if possible)
4. **Final**: Validate within bounds

### "Going" Videos

1. **Initial**: `duration = end_time - start_time` (selected to be closest to `ideal_duration`)
2. **If `duration > max_duration`**: 
   - If `ideal_duration <= max_duration`: Trim to `ideal_duration`
   - Otherwise: Trim to `max_duration`
3. **If `duration < min_duration`**: Extend backward (earlier) → `duration = min_duration` (if possible)
4. **Final**: Validate within bounds

---

## Technical Details

### Frame Sampling

- Samples frames at ~10 frames per second for efficiency
- Formula: `sample_interval = max(1, int(fps * 0.1))`
- Reduces processing time while maintaining detection accuracy

### Motion Detection Parameters (Coming)

- Background subtractor: MOG2 with `history=500`, `varThreshold=50`, `detectShadows=True`
- Minimum motion area: 1% of frame area
- Growth detection window: 5 samples (~0.5 seconds)
- Minimum growth factor: 2.0x (area must double to be considered rider)
- Motion stop threshold: 5% of maximum detected area
- Gap tolerance: 3 samples (~0.3 seconds)
- Extension after motion stop: 1.5 seconds

### Face Detection Parameters (Both)

- Model: MediaPipe Face Detection (full-range, `model_selection=1`)
- Confidence threshold: 0.5
- Minimum face size: 6% of frame width
- Stability requirement (Going): 3 consecutive detections
- Minimum time between detections: 0.5 seconds

---

## Return Values

### Valid Result

```json
{
  "input_video": "path/to/video.mp4",
  "direction": "coming" | "going",
  "start_time": 2.5,
  "end_time": 12.3,
  "duration": 9.8,
  "valid": true,
  "output_video": "path/to/output.mp4"  // if output_video_path was provided
}
```

### Invalid Result

```json
{
  "input_video": "path/to/video.mp4",
  "direction": "coming" | "going",
  "valid": false,
  "reason": "Detected duration 1.5s is below minimum 2.0s (even after extending)"
}
```

---

## Example Scenarios

### "Going" Video - Face Detected

- Video duration: 15 seconds
- Face detections at: 2s, 5s, 8s, 12s
- `ideal_duration = 8.0`, `min_duration = 2.0`, `max_duration = 20.0`

**Selection process**:
- Detection at 2s → duration = 13s (diff from ideal: 5s)
- Detection at 5s → duration = 10s (diff from ideal: 2s)
- Detection at 8s → duration = 7s (diff from ideal: 1s) ← **Best match**
- Detection at 12s → duration = 3s (diff from ideal: 5s)

**Result**: Start at 8s, end at 15s, duration = 7s

**Duration rules applied**:
- 7s < 8s ideal, but 7s >= 2s min → Valid
- No trimming needed (7s < 20s max)

### "Going" Video - No Face Detected

- Video duration: 10 seconds
- No face detections
- `ideal_duration = 8.0`, `min_duration = 2.0`, `max_duration = 20.0`

**Result**: Start at 0s, end at 8s (ideal_duration), duration = 8s

**Duration rules applied**:
- 8s >= 2s min → Valid
- 8s <= 20s max → Valid

### "Coming" Video - Motion Detected

- Video duration: 20 seconds
- Rider motion detected at: 5s
- Motion stops at: 18s
- `min_duration = 2.0`, `max_duration = 15.0`

**Initial calculation**:
- Start: 5s
- End: 18s + 1.5s = 19.5s (clamped to 20s)
- Duration: 15s

**Duration rules applied**:
- 15s <= 15s max → Valid (at limit)
- 15s >= 2s min → Valid

---

## Summary

The detection system uses different strategies for "coming" and "going" videos:

- **"Coming"**: Focuses on motion detection to identify when the rider enters the frame, with face detection used for correlation
- **"Going"**: Focuses on face detection to identify when the rider looks at the camera, selecting the detection that creates a clip closest to `ideal_duration`

Both directions apply duration constraints (`min_duration` and `max_duration`), but "going" videos also use `ideal_duration` as a target for optimal clip selection.
