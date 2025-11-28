# Photo Capture Logic

This document describes how `capture_images.py` evaluates frames, ranks candidates, and produces still images for every riding scenario. The script can run on its own, but it is most often triggered by `detect_zipline_segment(..., capture_images=True, ...)` so that trimming and photo export happen together.

---

## Parameters

`capture_images_from_video` accepts the following high-level arguments:

| Parameter | Description |
|-----------|-------------|
| `video_path` | Source video to process. |
| `mode` | `"going"`, `"coming"`, or `"group"`; controls which detectors run. Defaults to `"going"` when nothing is specified. |
| `min_pictures` / `max_pictures` | Bounds on how many frames are saved after ranking candidates (defaults to 5/10). |
| `platform_number` | Optional identifier echoed back inside the result (no longer changes behaviour—the detector relies on explicit parameters passed from `face-detection.py`). |
| `output_dir` | Destination folder; defaults to `<video_name>-images` next to the video. |
| `sharpness_threshold` | Frames below this Laplacian-variance value are skipped early (default 100). |
| `show_progress` | Prints per-5-second status updates. |
| `show_frames` | Opens an OpenCV window that overlays detections and debugging text. |
| `start_time` / `end_time` | Restrict processing to a trimmed segment (used by `detect_zipline_segment`). |

> **Extra captures**: When the detection pipeline passes `capture_offset_after_start` or `capture_offset_before_end`, `face-detection.py` saves additional frames (`extra_after_start_*`, `extra_before_end_*`) after the main capture run completes. Those files are appended to `image_capture["extra_captures"]`.

---

## Common Pipeline

1. **Initialization**
   - Opens the video once, determines FPS, total frames, and derives the guide-region size (used in “coming” mode).
   - Samples frames roughly every 0.1 s (`sample_interval = max(1, int(fps * 0.1))`) to keep inference real-time-friendly.
2. **Pre‑filters**
   - Converts the sampled frame to RGB for MediaPipe.
   - Rejects blurry frames before running expensive models.
3. **Mode-specific detectors**
   - Results in a list of candidate frames, each with a score and metadata used for overlays.
4. **Ranking & Export**
   - Sorts candidates by score descending.
   - Ensures we have at least `min_pictures`; otherwise the call returns `success=False` with a descriptive error.
   - Saves the top `min(max_pictures, len(candidate_frames))` frames using the naming pattern `frame_<frame_idx>_t<seconds>.jpg`.

---

## Mode Details

### Mode: `going`
Focused on portrait-quality face shots:

- Uses MediaPipe Face Detection (`model_selection=1`) plus Face Mesh for smile and frontal-orientation checks.
- Face must cover ≥ 6 % of the frame width and remain frontal; the pipeline discards side profiles.
- Score = facial confidence (50 %), sharpness (30 %), plus a smile bonus (20 %).
- The visualization window highlights smiles in green and lists frame time, sharpness, smile confidence, and aggregate score.

### Mode: `coming`
Captures riders approaching the camera while ignoring the guide standing at the launch platform:

- Background subtractor (MOG2) isolates moving blobs. Candidates must cover ≥ 2 % of the frame.
- The guide is filtered out using a configurable bottom-left “guide region”. We only keep detections whose bounding boxes are not fully inside that rectangle. Current defaults (also overridable per platform) are **40 %** of width and **80 %** of height.
- Optionally looks for faces inside the contour; detections with a visible face get a score boost.
- Score combines contour area, face bonus, and sharpness.
- The overlay shows the guide region border, whether a candidate was rejected as “guide only”, and the current score.

### Mode: `group`
Used for wide shots with multiple riders:

- Runs both person detection (background subtraction) and face detection in parallel.
- Counts people/faces and awards a bonus when both detectors agree (#people AND #faces > 0).
- Faces only need to span ≥ 3 % of the frame to handle distant subjects.
- Score = total count (main weight) + average face confidence + sharpness + agreement bonus.
- Only candidate frames with a `total_count` ≥ 2 are considered valid.

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

If the capture fails (e.g., not enough candidates), the payload sets `success=False` and includes a human-readable `error`.

---

## Real-time Visualization

Enabling `show_frames=True` opens a live OpenCV window:

- Draws bounding boxes (faces in green/red, people in orange).
- Displays current detection mode, candidate counts, sharpness, confidence, and accept/reject reasons.
- Press `q` to stop processing early (the function returns the current error state).

---

