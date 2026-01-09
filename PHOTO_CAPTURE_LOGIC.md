# Photo Capture Logic

This document describes how `capture_images.py` evaluates frames, ranks candidates, and produces still images for every riding scenario. The script can run on its own, but it is most often triggered by `detect_zipline_segment(..., capture_images=True, ...)` so that trimming and photo export happen together.

---

## Parameters

`capture_images_from_video` accepts the following high-level arguments:

| Parameter                       | Description                                                                                                                                                 |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `video_path`                    | Source video to process.                                                                                                                                    |
| `mode`                          | `"going"`, `"coming"`, or `"group"`; controls which detectors run. Defaults to `"going"` when nothing is specified.                                         |
| `min_pictures` / `max_pictures` | Bounds on how many frames are saved after ranking candidates (defaults to 5/10).                                                                            |
| `min_delay_seconds`             | Minimum time gap in seconds between any two captured images (default 2.0). Ensures photos are spread out over time.                                          |
| `platform_number`               | Optional identifier echoed back inside the result (no longer changes behaviour—the detector relies on explicit parameters passed from `face-detection.py`). |
| `output_dir`                    | Destination folder; defaults to `<video_name>-images` next to the video.                                                                                    |
| `sharpness_threshold`           | Frames below this Laplacian-variance value are skipped early (default 100).                                                                                 |
| `show_progress`                 | Prints per-5-second status updates.                                                                                                                         |
| `show_frames`                   | Opens an OpenCV window that overlays detections and debugging text.                                                                                         |
| `start_time` / `end_time`       | Restrict processing to a trimmed segment (used by `detect_zipline_segment`).                                                                                |

> **Extra captures**: When the detection pipeline passes `capture_offset_after_start` or `capture_offset_before_end`, `face-detection.py` saves additional frames (`extra_after_start_*`, `extra_before_end_*`) after the main capture run completes. Those files are appended to `image_capture["extra_captures"]`.

---

## Common Pipeline

1. **Initialization**
   - Opens the video once, determines FPS, total frames, and derives the guide-region size (used in "coming" mode).
   - Samples frames roughly every 0.1 s (`sample_interval = max(1, int(fps * 0.1))`) to keep inference real-time-friendly.

2. **Pre-filters**
   - Converts the sampled frame to RGB for MediaPipe.
   - Rejects blurry frames before running expensive models (frames below `sharpness_threshold` are skipped).

3. **Mode-specific detectors**
   - Each mode runs its own detection logic and assigns candidates a **priority tier** and a **score**.
   - Results in a list of candidate frames, each with metadata used for ranking and overlays.

4. **Ranking & Selection**
   - All candidates are sorted by `(priority tier, score)` descending (higher tier wins; within a tier, higher score wins).
   - Selection algorithm:
     - Starts with the highest priority tier and works downward.
     - For each candidate, checks if it's at least `min_delay_seconds` away from all already-selected frames.
     - Adds candidates that pass the delay check until `max_pictures` is reached.
   - Ensures we have at least `min_pictures`; otherwise the call returns `success=False` with a descriptive error.
   - Saves the selected frames using the naming pattern `frame_<frame_idx>_t<seconds>.jpg`.

---

## Mode Details

### Mode: `going`

Focused on capturing two distinct images: a portrait-quality face shot and a person-only shot of the rider going down the zip line.

**Detection Method:**
- Uses MediaPipe Face Detection (`model_selection=1`) plus Face Mesh for:
  - Frontal-orientation checks (face roughly looking at the camera).
  - Smile detection (based on mouth landmark geometry).
  - Eyes-open detection (lightweight eye-aspect-ratio style metric).
- Face must cover **≥ 6% of the frame width** to be considered valid.
- If no face is detected, falls back to person detection using background subtraction (lowest priority).

**Priority Tiers:**
- **Tier 4 (best)**: Clear **face**, **frontal**, **smiling**, **eyes open**.
- **Tier 3**: Clear **face**, **frontal** (eyes/smile may be neutral or low-confidence).
- **Tier 2**: Clear **face**, but **not frontal** (side/angled profile) that still passes size checks.
- **Tier 1 (fallback)**: **Person-only region** (coarse person blob from background subtraction) when no valid face candidate was available in that frame. This captures the rider going down the zip line when the face is no longer clearly visible.

**Scoring:**
- Base score from face detection confidence (50%) and sharpness (25%).
- Plus bonuses for **smile confidence** (15%) and **eyes-open confidence** (10%) when available.
- Person-only fallback uses area (40%) and sharpness (30%).

**Selection (Special Logic for "going" mode):**
- **Requires exactly 2 images**: 1 from tier 4/3/2 (face detection) AND 1 from tier 1 (person-only).
- **Face tier selection**: First selects the best face image (tier 4, 3, or 2) based on priority and score.
- **Tier 1 selection (time-based window)**:
  - Finds the time of the selected face tier image (e.g., at 0.33s).
  - Defines a **2-second window** starting 0.3 seconds after the face tier time (e.g., 0.63s to 2.33s).
  - Looks at **ALL candidates** (any tier) within this window, excluding the face tier candidate itself.
  - Selects the candidate with the **largest surface area** (closest person) in that window.
  - Ensures the tier 1 image is from a **different frame** than the face tier image (checked by `frame_count` and time).
  - **Fallback**: If no candidates found in the 2-second window, tries candidates around 1 second after the face tier time (with 0.3s tolerance), but still within the 2-second limit.
  - If still no candidates found, the capture fails with a descriptive error.
- **Time gap**: The tier 1 image is allowed to be close in time to the face tier image (within the 2-second window), so `min_delay_seconds` is not enforced between these two specific images.
- This ensures we capture both a clear face shot and a person-only shot of the rider going down the zip line, with the person shot taken shortly after the face shot when the rider is still relatively close to the camera.

---

### Mode: `coming`

Captures riders approaching the camera while ignoring the guide standing at the launch platform:

**Detection Method:**
- Background subtractor (MOG2) isolates moving blobs. Candidates must cover **≥ 2% of the frame**.
- The guide is filtered out using a configurable bottom-left "guide region". We only keep detections whose bounding boxes are **not fully inside** that rectangle. Current defaults are **40% of width** and **80% of height**.
- Optionally looks for faces inside the detected person region; detections with a visible face get a significant score boost.
- For person-only detections (no face), the rider must be at least **12% of frame width** to avoid capturing very distant shots.

**Priority Tiers:**
- **Tier 3 (best)**: Person with at least one **frontal face** outside the guide region.
- **Tier 2**: Person with at least one **non-frontal face** outside the guide region.
- **Tier 1**: Person-only blob (no face, but still a valid moving rider outside the guide region and large enough).

**Scoring:**
- Base score from person area (50%) and sharpness (20%).
- Face bonuses:
  - **Frontal face**: +0.7 bonus (Tier 3).
  - **Non-frontal face**: +0.4 bonus (Tier 2).
- Person-only candidates (Tier 1) use area and sharpness only.

**Selection:**
- Images are selected based on **priority tier first**, then **score within tier**.
- Must be at least `min_delay_seconds` apart from other selected images.
- The overlay shows the guide region border, whether a candidate was rejected as "guide only" or "too small", and the current score.

---

### Mode: `group`

Used for wide shots with multiple riders:

**Detection Method:**
- Runs both person detection (background subtraction) and face detection in parallel.
- Person detection: Background subtractor finds moving blobs covering **≥ 2% of the frame**.
- Face detection: MediaPipe finds faces that span **≥ 3% of the frame width** (lower threshold than "going" mode to handle distant subjects).
- A frame is only considered a **group** candidate if there are **2 or more people/faces in total**; a single person or a single face alone is never enough.

**Priority Tiers:**
- **Tier 3 (best)**: Combined detection of **2+ riders** where both face and person detectors fire (e.g., several people plus visible faces).
- **Tier 2**: **2+ faces** detected, but person blobs are weak or missing.
- **Tier 1**: **2+ people** detected from the person blobs, but no reliable faces.

**Scoring:**
- Count score: Total number of people/faces × 0.4 (main weight).
- Average face confidence × 0.2 (if faces are detected).
- Sharpness component × 0.2.
- Detection method bonus: +0.2 when both person and face detectors agree (both find people).
- Uses the maximum of person count and face count, with a small bonus (+0.5) when both methods agree.

**Selection:**
- Images are selected based on **priority tier first**, then **score within tier**.
- Must be at least `min_delay_seconds` apart from other selected images.
- Higher tiers are always preferred, ensuring we get frames where both detection methods confirm multiple people.

---

## Output Structure

Example success payload:

```json
{
  "success": true,
  "images_captured": 5,
  "candidates_found": 14,
  "output_dir": "going-1-images",
  "captured_files": [
    "going-1-images/frame_000070_t2.34s.jpg",
    "going-1-images/frame_000072_t2.40s.jpg",
    "going-1-images/frame_000074_t2.47s.jpg",
    "going-1-images/frame_000120_t4.00s.jpg",
    "going-1-images/frame_000150_t5.00s.jpg"
  ],
  "mode": "going",
  "platform_number": 1,
  "extra_captures": [
    "going-1-images/extra_after_start_2.00_t4.34s.jpg",
    "going-1-images/extra_before_end_2.00_t6.17s.jpg"
  ]
}
```

If the capture fails (e.g., not enough candidates or not enough candidates that satisfy `min_delay_seconds`), the payload sets `success=False` and includes a human-readable `error`.

---

## Real-time Visualization

Enabling `show_frames=True` opens a live OpenCV window:

- **Going mode**: Draws face bounding boxes (green for smiling, red for non-smiling), person fallback boxes (orange), and displays tier, confidence, smile status, eyes status, and sharpness.
- **Coming mode**: Draws the guide region border (red), person bounding boxes (green if accepted, red if filtered), face detections (green for frontal, yellow for non-frontal), and shows accept/reject reasons.
- **Group mode**: Draws all detected people (orange boxes) and all detected faces (green boxes), and displays counts, confidence, and tier information.
- Displays current detection mode, candidate counts, sharpness, confidence, and accept/reject reasons.
- Press `q` to stop processing early (the function returns the current error state).

---
