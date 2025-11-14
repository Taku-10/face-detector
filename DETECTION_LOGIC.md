# Face Detection Logic Documentation

This document explains the detection logic, start/end time determination, and duration rules for both "coming" and "going" video directions.

---

## Parameters

### Core Parameters

- **`min_duration`**: Minimum duration for a valid clip (default: 4.0 seconds)
- **`max_duration`**: Maximum duration cap (optional, can be `None`)
- **`ideal_duration`**: Ideal duration for clips (default: 8.0 seconds)
  - Used for both "coming" and "going" videos to select the best detection
  - For "coming": Picks motion detection that creates clip closest to ideal_duration
  - For "going": Picks face detection that creates clip closest to ideal_duration
  - Ensures clips are close to the desired length when possible
- **`direction`**: Either "coming" or "going" (direction of rider)
- **`end_trim_seconds`**: Seconds to remove from end for "coming" videos when end reaches video end (default: 1.0)
  - Only applies to "coming" videos when `(start_time + ideal_duration) >= video_duration`
  - Used to crop out guide's hand switching off camera
  - Can be configured per platform or as a direct parameter
- **`backward_extension_seconds`**: Seconds to extend backward (earlier) when duration is too short (default: 2.0)
  - For "coming" videos: Used first when `duration < min_duration` (extends start time earlier)
  - For "going" videos: Used as fallback when forward extension fails (extends start time earlier)
  - Can be configured per platform or as a direct parameter
- **`platform_number`**: Platform number (1, 2, 3, etc.) to use platform-specific settings
  - If provided, automatically applies platform's direction, min_duration, max_duration, ideal_duration, end_trim_seconds, and backward_extension_seconds
  - Individual parameters can still override platform settings if explicitly provided

### Platform Configuration

Each platform can have its own customized settings defined in `PLATFORM_CONFIGS`:

```python
PLATFORM_CONFIGS = {
    1: {
        "direction": "going",
        "min_duration": 5.0,
        "max_duration": 10.0,
        "ideal_duration": 8.0,
        "end_trim_seconds": 0.0,  # Seconds to remove from end for "coming" videos
        "backward_extension_seconds": 2.0,  # Seconds to extend backward when duration is too short
    },
    2: {
        "direction": "coming",
        "min_duration": 5.0,
        "max_duration": 10.0,
        "ideal_duration": 8.0,
        "end_trim_seconds": 1.0,  # Seconds to remove from end for "coming" videos
        "backward_extension_seconds": 2.0,  # Seconds to extend backward when duration is too short
    },
    3: {
        "direction": "coming",
        "min_duration": 5.0,
        "max_duration": 15.0,
        "ideal_duration": 10.0,
        "end_trim_seconds": 2.0,  # Example: Platform 3 removes 2 seconds
        "backward_extension_seconds": 3.0,  # Example: Platform 3 extends backward by 3 seconds
    },
    # Add more platforms as needed
}
```

**Usage Examples**:

1. **Use platform configuration** (recommended):

   ```python
   result = detect_zipline_segment(
       input_video_path="video.mp4",
       platform_number=2,  # Uses platform 2's settings
   )
   ```

2. **Override platform settings**:

   ```python
   result = detect_zipline_segment(
       input_video_path="video.mp4",
       platform_number=2,  # Base settings from platform 2
       ideal_duration=10.0,  # Override ideal_duration
   )
   ```

3. **Use explicit parameters** (no platform):
   ```python
   result = detect_zipline_segment(
       input_video_path="video.mp4",
       direction="coming",
       min_duration=2.0,
       max_duration=20.0,
       ideal_duration=8.0,
       end_trim_seconds=2.0,  # Remove 2 seconds from end for "coming" videos
       backward_extension_seconds=3.0,  # Extend backward by 3 seconds when duration is too short
   )
   ```

**Parameter Priority**:

1. Explicit parameters (if provided) override platform settings
2. Platform settings (if `platform_number` provided)
3. Default values (if neither platform nor explicit parameters)

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

The start time is determined by detecting when the **rider first enters the frame** (even from far away), while filtering out the guide's constant motion:

1. **Collect motion samples** throughout the video

   - Each sample contains: time, has_motion flag, motion area
   - Motion area is calculated from the largest contour
   - Minimum motion threshold: 1% of frame area

2. **Filter out guide's constant motion**:

   - Analyze first 10% of video to identify guide's constant small motion
   - Calculate average early motion area to use as a baseline
   - Skip any motion that is similar to early constant motion (within 30% variance)

3. **Identify rider's motion** (when rider first enters frame):

   - **Primary method**: Look for motion that **grows significantly** over time

     - Growth window: 5 samples (~0.5 seconds)
     - Growth factor: Motion area must grow by at least 1.5x (lowered to catch motion earlier)
     - This identifies the rider approaching (motion grows) vs. guide (constant small motion)
     - Skips motion similar to early constant guide motion

   - **Secondary method**: Find first **NEW motion** that appears

     - Motion that appears after a period of no motion, OR
     - Motion that is significantly different from early constant motion (not similar to guide)
     - Motion that is preceded by no motion or much smaller motion (30% increase)
     - This catches when rider first enters frame, even from far away

   - **Final fallback**: Use first motion that's at least 5% of maximum detected area
     - Very permissive threshold to catch even small initial motion
     - Ensures we don't miss the rider entering from far away

4. **Set start time**:
   - Start time = Time of first detected rider motion (when rider enters frame)
   - If no rider motion detected: Fallback to `video_duration - ideal_duration`

**Result**: Start time = When rider first enters frame (even from far away), filtered from guide's constant motion

### End Time Logic

The end time is calculated based on `ideal_duration`:

1. **Calculate initial end time**:

   - Formula: `end_time = start_time + ideal_duration`
   - This creates a clip of exactly `ideal_duration` length from the detected start

2. **Apply end_trim_seconds** (after all duration constraints are applied):
   - **If `end_time >= video_duration - 1.0`** (within 1 second of video end):
     - Set `end_time = video_duration - end_trim_seconds` (removes `end_trim_seconds` to crop out guide's hand switching off camera)
     - This ensures we crop the guide's hand even if we're close to but not exactly at video end
     - `end_trim_seconds` is configurable per platform or as a direct parameter (default: 1.0 seconds)
   - **If `end_time < video_duration - 1.0`**:
     - End time is more than 1 second before video end, use calculated end time as is

**Result**: End time = `start_time + ideal_duration`, then adjusted to `video_duration - end_trim_seconds` if within 1 second of video end (after all duration constraints are applied)

**Note**: The `end_trim_seconds` is applied as a final step after all duration constraints (max_duration, min_duration, forward/backward extensions) have been applied. This ensures the guide's hand is cropped even if the end time is close to but not exactly at the video end.

### Duration Rules for "Coming" Videos

After calculating initial start and end times (using `ideal_duration`), the following rules are applied **in order**:

#### Rule 1: Apply max_duration constraint (if set)

- **If `duration > max_duration`**:
  - Trim from the **end** to reach `max_duration`
  - Formula: `end_time = start_time + max_duration`
  - `duration = max_duration`

#### Rule 2: Apply min_duration constraint

- **If `duration < min_duration`**:

  - **Step 1**: Try to extend **backward** (earlier in video) by `backward_extension_seconds`

    - Calculate available backward time: `available_backward = start_time`
    - **If `available_backward >= backward_extension_seconds`**:
      - Extend start time backward: `start_time = start_time - backward_extension_seconds`
      - `duration = end_time - start_time`
      - If still too short after backward extension, proceed to Step 2
    - **If `available_backward < backward_extension_seconds`**:
      - Cannot extend backward enough, proceed to Step 2

  - **Step 2**: If still too short after backward extension (or backward extension not possible), extend **forward** (later) by 2 seconds
    - Calculate available forward time: `available_forward = video_duration - end_time`
    - **If `available_forward >= 2.0`**:
      - Extend end time forward by 2 seconds: `end_time = end_time + 2.0`
      - `duration = end_time - start_time`
      - If still too short after forward extension: Return invalid result
    - **If `available_forward < 2.0`**:
      - Extend as much as possible: `end_time = video_duration`
      - `duration = end_time - start_time`
      - If still too short: Return invalid result

#### Rule 3: Final validation

- Ensures final duration is within `min_duration` and `max_duration` bounds
- If `max_duration` is set and duration still exceeds it: Final trim to `max_duration`
- If valid, returns the segment with rounded times (2 decimal places)

**Key Points**:

- `ideal_duration` determines the initial end time: `end_time = start_time + ideal_duration`
- After all duration constraints are applied, if `end_time >= video_duration - 1.0` (within 1 second of video end), it's set to `video_duration - end_trim_seconds` to crop guide's hand
- This ensures the guide's hand is cropped even if the end time is close to but not exactly at the video end
- `end_trim_seconds` is configurable per platform or as a direct parameter (default: 1.0 seconds)
- `backward_extension_seconds` is configurable per platform or as a direct parameter (default: 2.0 seconds)
- If duration exceeds `max_duration`, trims to `max_duration`
- If duration is below `min_duration`, extends backward first (by `backward_extension_seconds`), then forward (by 2 seconds) if needed

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

  - **Step 1**: Try to extend **forward** (later in video) by 2 seconds

    - Calculate available forward time: `available_forward = video_duration - end_time`
    - **If `available_forward >= 2.0`**:
      - Extend end time forward by 2 seconds: `end_time = end_time + 2.0`
      - `duration = end_time - start_time`
      - If still too short after forward extension, proceed to Step 2
    - **If `available_forward < 2.0`**:
      - Extend as much as possible: `end_time = video_duration`
      - `duration = end_time - start_time`
      - If still too short, proceed to Step 2

  - **Step 2**: If still too short after forward extension (or forward extension not possible), extend **backward** (earlier) by `backward_extension_seconds`
    - Calculate available backward time: `available_backward = start_time`
    - **If `available_backward >= backward_extension_seconds`**:
      - Extend start time backward: `start_time = start_time - backward_extension_seconds`
      - `duration = end_time - start_time`
      - If still too short after backward extension: Return invalid result
    - **If `available_backward < backward_extension_seconds`**:
      - Extend as much as possible: `start_time = 0.0`
      - `duration = end_time - start_time`
      - If still too short: Return invalid result

#### Rule 3: Final validation

- Ensures final duration is within `min_duration` and `max_duration` bounds
- If `max_duration` is set and duration still exceeds it: Final trim to `max_duration`
- If valid, returns the segment with rounded times (2 decimal places)

**Key Points**:

- `ideal_duration` is the **primary target** for "going" videos
- System selects face detection that gets closest to `ideal_duration`
- `backward_extension_seconds` is configurable per platform or as a direct parameter (default: 2.0 seconds)
- If duration exceeds `max_duration`, trims to `ideal_duration` (if within max) or `max_duration`
- If duration is below `min_duration`, extends forward first (by 2 seconds), then backward (by `backward_extension_seconds`) if needed

---

## Comparison Table

| Aspect                    | Coming                                                                                       | Going                                       |
| ------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Start Detection**       | First motion detection (rider entering)                                                      | Face detection (optimal for ideal_duration) |
| **End Detection**         | `start_time + ideal_duration` (then `video_duration - end_trim_seconds` if within 1s of end) | Video end (or ideal_duration if no faces)   |
| **Primary Method**        | Motion detection                                                                             | Face detection only                         |
| **Filtering**             | Filters guide's constant motion                                                              | Filters small/weak detections               |
| **Extension**             | N/A (uses ideal_duration directly)                                                           | N/A                                         |
| **Duration Target**       | `ideal_duration` (exact)                                                                     | Closest to ideal_duration                   |
| **Uses ideal_duration**   | ✅ Yes (for end time calculation)                                                            | ✅ Yes (for start time selection)           |
| **No Detection Fallback** | `(video_duration - ideal_duration) to (video_duration - end_trim_seconds)`                   | `0 to ideal_duration`                       |
| **Special Handling**      | Removes `end_trim_seconds` at end if reaches video end (crop guide's hand, configurable)     | N/A                                         |

---

## Duration Rules Summary

### "Coming" Videos

1. **Initial**:
   - Start: First motion detection (rider entering frame)
   - End: `start_time + ideal_duration`
   - `duration = end_time - start_time`
2. **If `duration > max_duration`**: Trim from end → `duration = max_duration`
3. **If `duration < min_duration`**:
   - First: Extend backward (earlier) by `backward_extension_seconds` → `start_time = start_time - backward_extension_seconds` (if possible)
   - If still too short: Extend forward (later) by 2 seconds → `end_time = end_time + 2.0` (if possible)
4. **Apply end_trim_seconds**: If `end_time >= video_duration - 1.0` (within 1 second of video end) → `end_time = video_duration - end_trim_seconds` (crop guide's hand)
5. **Final**: Validate within bounds

### "Going" Videos

1. **Initial**: `duration = end_time - start_time` (selected to be closest to `ideal_duration`)
   - System collects all face detections
   - For each detection, calculates duration (video_end - detection_time)
   - Selects the detection that creates duration closest to `ideal_duration`
2. **If `duration > max_duration`**:
   - If `ideal_duration <= max_duration`: Trim from end to `ideal_duration`
   - Otherwise: Trim to `max_duration`
3. **If `duration < min_duration`**:
   - First: Extend forward (later) by 2 seconds → `end_time = end_time + 2.0` (if possible)
   - If still too short: Extend backward (earlier) by `backward_extension_seconds` → `start_time = start_time - backward_extension_seconds` (if possible)
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
  "output_video": "path/to/output.mp4",  // if output_video_path was provided
  "trimmed_video": "path/to/trimmed.mp4",  // if trim_output_path was provided
  "platform_number": 2  // if platform_number was provided
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

### "Coming" Video - Motion Detected (Case 1: End Before Video End)

- Video duration: 20 seconds
- Rider motion detected at: 5s
- `ideal_duration = 8.0`, `min_duration = 2.0`, `max_duration = 15.0`

**Initial calculation**:

- Start: 5s
- Calculated end: 5s + 8s = 13s
- Since 13s < 20s (video end): End = 13s
- Duration: 8s

**Duration rules applied**:

- 8s <= 15s max → Valid
- 8s >= 2s min → Valid

### "Coming" Video - Motion Detected (Case 2: End Reaches Video End)

- Video duration: 15 seconds
- Rider motion detected at: 8s
- `ideal_duration = 8.0`, `min_duration = 2.0`, `max_duration = 15.0`, `end_trim_seconds = 1.0`

**Initial calculation**:

- Start: 8s
- Calculated end: 8s + 8s = 16s
- Since 16s >= 15s (video end): End = 15s - 1.0 = 14s (removes `end_trim_seconds` to crop guide's hand)
- Duration: 6s

**Duration rules applied**:

- 6s <= 15s max → Valid
- 6s >= 2s min → Valid

### "Coming" Video - Motion Detected (Case 3: Custom end_trim_seconds)

- Video duration: 15 seconds
- Rider motion detected at: 7s
- `ideal_duration = 8.0`, `min_duration = 2.0`, `max_duration = 15.0`, `end_trim_seconds = 2.0` (platform-specific)

**Initial calculation**:

- Start: 7s
- Calculated end: 7s + 8s = 15s
- Since 15s >= 15s (video end): End = 15s - 2.0 = 13s (removes 2s as configured)
- Duration: 6s

**Duration rules applied**:

- 6s <= 15s max → Valid
- 6s >= 2s min → Valid

### "Coming" Video - Motion Detected (Case 4: Backward Extension Needed)

- Video duration: 12 seconds
- Rider motion detected at: 10s
- `ideal_duration = 8.0`, `min_duration = 5.0`, `max_duration = 15.0`, `backward_extension_seconds = 2.0`, `end_trim_seconds = 1.0`

**Initial calculation**:

- Start: 10s
- Calculated end: 10s + 8s = 18s
- Since 18s >= 12s (video end): End = 12s - 1.0 = 11s (removes `end_trim_seconds` to crop guide's hand)
- Duration: 1s (11s - 10s)

**Duration rules applied**:

- 1s < 5s min → Need to extend
- **Step 1 - Backward Extension**:
  - Available backward: 10s
  - Extend backward by 2s: Start = 10s - 2s = 8s
  - New duration: 11s - 8s = 3s
  - Still too short (3s < 5s min)
- **Step 2 - Forward Extension**:
  - Available forward: 12s - 11s = 1s
  - Can't extend 2s forward, extend as much as possible: End = 12s
  - New duration: 12s - 8s = 4s
  - Still too short (4s < 5s min)
- **Result**: Invalid (duration 4s is below minimum 5s even after extending backward and forward)

### "Coming" Video - Motion Detected (Case 5: Backward Extension Success)

- Video duration: 12 seconds
- Rider motion detected at: 9s
- `ideal_duration = 8.0`, `min_duration = 5.0`, `max_duration = 15.0`, `backward_extension_seconds = 2.0`, `end_trim_seconds = 1.0`

**Initial calculation**:

- Start: 9s
- Calculated end: 9s + 8s = 17s
- Since 17s >= 12s (video end): End = 12s - 1.0 = 11s (removes `end_trim_seconds` to crop guide's hand)
- Duration: 2s (11s - 9s)

**Duration rules applied**:

- 2s < 5s min → Need to extend
- **Step 1 - Backward Extension**:
  - Available backward: 9s
  - Extend backward by 2s: Start = 9s - 2s = 7s
  - New duration: 11s - 7s = 4s
  - Still too short (4s < 5s min)
- **Step 2 - Forward Extension**:
  - Available forward: 12s - 11s = 1s
  - Can't extend 2s forward, extend as much as possible: End = 12s
  - New duration: 12s - 7s = 5s
  - ✅ Meets minimum (5s >= 5s min)
- **Final**: Valid (duration = 5s)

### "Going" Video - Face Detected (Case 1: Forward Extension Needed)

- Video duration: 10 seconds
- Face detected at: 8s
- `ideal_duration = 8.0`, `min_duration = 5.0`, `max_duration = 15.0`, `backward_extension_seconds = 2.0`

**Initial calculation**:

- Start: 8s (face detection)
- End: 10s (video end)
- Duration: 2s (10s - 8s)

**Duration rules applied**:

- 2s < 5s min → Need to extend
- **Step 1 - Forward Extension**:
  - Available forward: 10s - 10s = 0s
  - Can't extend forward (already at video end)
- **Step 2 - Backward Extension**:
  - Available backward: 8s
  - Extend backward by 2s: Start = 8s - 2s = 6s
  - New duration: 10s - 6s = 4s
  - Still too short (4s < 5s min)
- **Result**: Invalid (duration 4s is below minimum 5s even after extending forward and backward)

### "Going" Video - Face Detected (Case 2: Forward and Backward Extension Success)

- Video duration: 12 seconds
- Face detected at: 10s
- `ideal_duration = 8.0`, `min_duration = 5.0`, `max_duration = 15.0`, `backward_extension_seconds = 2.0`

**Initial calculation**:

- Start: 10s (face detection)
- End: 12s (video end)
- Duration: 2s (12s - 10s)

**Duration rules applied**:

- 2s < 5s min → Need to extend
- **Step 1 - Forward Extension**:
  - Available forward: 12s - 12s = 0s
  - Can't extend forward (already at video end)
- **Step 2 - Backward Extension**:
  - Available backward: 10s
  - Extend backward by 2s: Start = 10s - 2s = 8s
  - New duration: 12s - 8s = 4s
  - Still too short (4s < 5s min)
  - Extend backward as much as possible: Start = 0s
  - New duration: 12s - 0s = 12s
  - ✅ Meets minimum (12s >= 5s min)
- **Final**: Valid (duration = 12s)

### "Going" Video - Face Detected (Case 3: Forward Extension Success)

- Video duration: 15 seconds
- Face detected at: 12s
- `ideal_duration = 8.0`, `min_duration = 5.0`, `max_duration = 15.0`, `backward_extension_seconds = 2.0`

**Initial calculation**:

- Start: 12s (face detection)
- End: 15s (video end)
- Duration: 3s (15s - 12s)

**Duration rules applied**:

- 3s < 5s min → Need to extend
- **Step 1 - Forward Extension**:
  - Available forward: 15s - 15s = 0s
  - Can't extend forward (already at video end)
- **Step 2 - Backward Extension**:
  - Available backward: 12s
  - Extend backward by 2s: Start = 12s - 2s = 10s
  - New duration: 15s - 10s = 5s
  - ✅ Meets minimum (5s >= 5s min)
- **Final**: Valid (duration = 5s)

### "Going" Video - Face Detected (Case 4: Forward Extension First, Then Backward)

- Video duration: 12 seconds
- Face detected at: 9s
- `ideal_duration = 8.0`, `min_duration = 5.0`, `max_duration = 15.0`, `backward_extension_seconds = 2.0`

**Initial calculation**:

- Start: 9s (face detection)
- End: 12s (video end)
- Duration: 3s (12s - 9s)

**Duration rules applied**:

- 3s < 5s min → Need to extend
- **Step 1 - Forward Extension**:
  - Available forward: 12s - 12s = 0s
  - Can't extend forward (already at video end)
- **Step 2 - Backward Extension**:
  - Available backward: 9s
  - Extend backward by 2s: Start = 9s - 2s = 7s
  - New duration: 12s - 7s = 5s
  - ✅ Meets minimum (5s >= 5s min)
- **Final**: Valid (duration = 5s)

### "Going" Video - Face Detected (Case 5: Forward Extension Works First)

- Video duration: 15 seconds
- Face detected at: 11s
- `ideal_duration = 8.0`, `min_duration = 5.0`, `max_duration = 15.0`, `backward_extension_seconds = 2.0`

**Initial calculation**:

- Start: 11s (face detection)
- End: 15s (video end)
- Duration: 4s (15s - 11s)

**Duration rules applied**:

- 4s < 5s min → Need to extend
- **Step 1 - Forward Extension**:
  - Available forward: 15s - 15s = 0s
  - Can't extend forward (already at video end)
- **Step 2 - Backward Extension**:
  - Available backward: 11s
  - Extend backward by 2s: Start = 11s - 2s = 9s
  - New duration: 15s - 9s = 6s
  - ✅ Meets minimum (6s >= 5s min)
- **Final**: Valid (duration = 6s)

---

## Summary

The detection system uses different strategies for "coming" and "going" videos:

- **"Coming"**:
  - Detects when rider enters frame using motion detection
  - End time = `start_time + ideal_duration`
  - After all duration constraints: If end time is within 1 second of video end, removes `end_trim_seconds` (configurable, default: 1.0s) to crop guide's hand switching off camera
  - Uses `ideal_duration` to determine exact end time from detected start
- **"Going"**:
  - Detects when rider looks at camera using face detection
  - Selects face detection that creates clip closest to `ideal_duration`
  - End time is always video end (or `ideal_duration` if no faces detected)

Both directions:

- Use `ideal_duration` as a target for clip length
- Apply duration constraints (`min_duration` and `max_duration`)
- Can automatically trim videos using detected start/end times
