# Photo Capture Logic

This document describes how `capture_images.py` evaluates frames, assigns **tiers**, ranks candidates, and exports still images. In production it is usually triggered by `face-detection.py` (after a segment is detected), but you can also run `capture_images.py` directly.

---

## Where capture runs (direction → capture mode)

`face-detection.py` has a **detection direction** (e.g. `"going"`, `"coming"`, `"walking"`, `"kitting"`, `"sitting"`) and a separate **photo capture mode** (the `mode=` argument passed into `capture_images_from_video`).

Default mapping (unless `capture_images_mode` overrides it):

- **`direction="coming"`** → `mode="coming"`
- **`direction="going"`** → `mode="going"`
- **`direction in ["walking", "kitting", "sitting"]`** → `mode="group"`

Overrides:

- **`capture_images_mode`** can explicitly be `"going"`, `"coming"`, `"group"`, or `"bridge"`.
- **Special case**: if **`direction=="walking"` and `capture_images_mode=="bridge"`**, `face-detection.py` deliberately passes `start_time=None` and `end_time=None` so that `bridge` mode can decide its own “tail-of-video” window (instead of using the detected trim).

---

## What we currently capture per platform (defaults)

These are the current defaults in `face-detection.py` (`PLATFORM_CONFIGS`), i.e. what happens when you call `detect_zipline_segment(..., platform_number=N)` without overriding capture parameters:

| Platform | Detection `direction` | Capture `mode` | `min/max` | Notes |
| --- | --- | --- | --- | --- |
| 1 | `going` | `going` | 1 / 2 | Captures up to 2 frames total, then tries offset-based extras (only if slots remain). |
| 2 | `coming` | `coming` | 1 / 1 | Intentionally selects the single best close rider image. |
| 3 | `walking` | `bridge` | 1 / 1 | Uses the `walking + bridge` special case: capture scans the tail of the full video (not the detected trim). |
| 4 | `kitting` | `group` | 1 / 5 | Captures multiple “best group” frames spread out by `min_delay_seconds`. |
| 5 | `sitting` | `group` | 1 / 5 | Same as kitting (group capture). |

How that maps to tiers (practically):

- **Platform 1 (`going`, max=2)**: you typically end up with **1 best face frame** (Tier 4/3/2) plus a second-best candidate that is often another good face frame, but can be **Tier 1** when faces are missing/unstable. The exact tier mix depends on what candidates exist and the `min_delay_seconds` spacing.
- **Platform 2 (`coming`, max=1)**: you get **one close/large rider** frame (Tier 3/2/1), with Tier 0 (obstructions) filtered out.
- **Platform 3 (`bridge`, max=1)**: you get **one end-window bridge** frame, preferring Tier 4/3/2, otherwise Tier 1.
- **Platforms 4–5 (`group`, max=5)**: you get up to 5 frames, preferring Tier 3, then 2, then 1, spaced by `min_delay_seconds`.

---

## Parameters

`capture_images_from_video(...)` (in `capture_images.py`) accepts the following high-level arguments:

| Parameter | Description |
| --- | --- |
| `video_path` | Source video to process. |
| `mode` | `"going"`, `"coming"`, `"group"`, or `"bridge"`. Defaults to `"going"` when not specified. |
| `min_pictures` / `max_pictures` | Bounds on how many frames are saved after ranking/selection (defaults to 5/10). |
| `min_delay_seconds` | Minimum time gap (seconds) enforced between any two selected frames (default 2.0). |
| `output_dir` | Destination folder; defaults to `<video_name>-images` next to the video. |
| `filename_prefix` | Optional prefix added to saved filenames (useful when multiple capture passes share the same `output_dir`). |
| `sharpness_threshold` | Blurry-frame cutoff used by the prefilter (default 100). |
| `show_progress` | Prints debug/progress information during capture. |
| `show_frames` | Opens an OpenCV window with overlays for debugging. |
| `start_time` / `end_time` | Restrict processing to a trimmed segment. (`face-detection.py` usually passes the detected segment, except the `walking + bridge` special case.) |
| `detection_area` | Optional ROI restriction (rectangle or polygon, normalized or pixel coords). Candidates outside this area are rejected. |
| `platform_number` | Optional identifier carried through to the result (does not change capture behavior). |

> **Extra captures (offsets)**: When `face-detection.py` passes `capture_offset_after_start` and/or `capture_offset_before_end`, it saves additional frames (`extra_after_start_*`, `extra_before_end_*`) after the main capture completes. Those filenames are appended to `image_capture["extra_captures"]` (subject to `capture_images_max` remaining slots).

---

## Common Pipeline (all modes)

1. **Initialization**
   - Opens video, reads FPS/shape, sets `sample_interval ≈ 0.1s`.
   - Initializes MediaPipe detectors (face detection + face mesh) and a background subtractor (more sensitive settings for `coming` / `bridge`).

2. **Time-window gating (mode-specific)**
   - **`coming`**: processes the full provided `[start_time, end_time]` segment; no extra half-segment or end-trim logic (we rely on `detection_area` instead).
   - **`bridge`**:
     - warms up the background subtractor for ~**0.7s** before accepting candidates
     - only considers the **last 5 seconds** of the region being scanned (if `start_time/end_time` not provided, that region is the full video)

3. **Pre-filters**
   - Rejects blurry frames early using `sharpness_threshold`.

4. **Candidate generation**
   - Each accepted candidate frame gets:
     - a **priority tier** (`priority_level`)
     - a **score** (mode-specific)
     - metadata like `time`, `frame_count`, `bbox`, `area`, face attributes, etc.

5. **Ranking & Selection (core algorithm)**
   - **Filter**: for `bridge`, Tier **0** candidates (obstructions) are removed before sorting/selection.
   - **Sort** (mode-specific emphasis):
     - **`coming`**: area dominates (closest rider), then tier, then score.
     - **`bridge`**: tier dominates, then area (closest walker), then score.
     - **`going` / `group`**: tier dominates, then score, then area.
   - **Tier-by-tier fill**:
     - group candidates by tier, iterate tiers from highest to lowest
     - within each tier, iterate candidates (already sorted for that tier)
     - add a candidate only if it is at least `min_delay_seconds` away from all already-selected frames
     - stop when `max_pictures` is reached
   - **Coming special case**: if `mode=="coming"` and `max_pictures==1`, the algorithm selects the single best (largest/closest) candidate without applying timing constraints.
   - **Bridge end-focus**:
     - after candidates are grouped by tier, bridge mode window-restricts each tier to the **last ~1 second** of candidate times (if that tier has any in-window candidates), so the selected image(s) tend to be from when the walker is closest.

---

## Mode Details (tiers + what they mean)

### Mode: `going`

Goal: capture the best-looking face frames when available, with a low-priority person fallback when faces aren’t detected.

- **Detection**
  - MediaPipe Face Detection + Face Mesh:
    - frontal check (looking at camera)
    - smile detection
    - eyes-open detection
  - Face must be **≥ 4%** of the **effective width**:
    - if a `detection_area` is configured, we use the **width of that area**
    - otherwise we use the full frame width.
  - If no face candidate is accepted for that sampled frame, a background-subtractor “person blob” is used as a fallback candidate.

- **Tiers**
  - **Tier 4**: face + frontal + smiling + eyes open
  - **Tier 3**: face + frontal (smile/eyes may be neutral)
  - **Tier 2**: face detected but not frontal
  - **Tier 1**: person-only fallback (no face)

- **Scoring**
  - **Face candidates**: \(0.5 \cdot \text{face_conf} + 0.25 \cdot \text{sharpness}\) plus smile/eyes bonuses (\(+0.15\), \(+0.1\)).
  - **Person fallback**: area-weighted + sharpness-weighted (area is the main proxy for “closeness”).

- **Selection**
  - Uses the common tier-by-tier selection algorithm (no “exactly 2 images” rule in code—how many you get depends on `min_pictures/max_pictures` passed in by the caller).

---

### Mode: `coming`

Goal: capture the rider **closest to the camera** while filtering the guide zone and rejecting obstructions near the end.

- **Detection**
  - Primary: background subtraction person detection (largest contours).
  - Uses face detection as a helper signal to distinguish guide/rider:
    - If the detected person is in the guide region, it can still be accepted when **2+ faces** are present (guide + rider), or when the person blob is **very large** (rider likely present even if face is hard to detect).
  - Time gating: only scans the **second half** of the segment; also trims `end_time` by **1 second** (when a segment is provided).

- **Tiers**
  - **Tier 3**: person + frontal face
  - **Tier 2**: person + non-frontal face
  - **Tier 1**: person only (no face)
  - **Tier 0**: obstruction (hand/object blocking camera) — filtered out before selection

- **Scoring**
  - Base: area component (closeness proxy) + sharpness.
  - Face bonus: +0.7 for frontal face, +0.4 for non-frontal face.
  - Obstructions get a heavily reduced score.

- **Selection**
  - Sort emphasizes **area first** (closest rider wins).
  - Filters out candidates that are too small relative to the best (keeps selection focused on close riders).
  - Special case: if `max_pictures==1`, selects the single best image regardless of `min_delay_seconds`.

---

### Mode: `group`

Goal: capture frames where **2+ people** are present (walking/kitting/sitting wide shots).

- **Detection**
  - Person blobs via background subtraction.
  - Faces via MediaPipe face detection (lower face-size threshold than `going` so distant faces can count).
  - A frame is only a candidate when total count indicates a group (**≥ 2**).

- **Tiers**
  - **Tier 3**: both person detection and face detection agree on a group (best).
  - **Tier 2**: 2+ faces detected (but weak/missing person blobs).
  - **Tier 1**: 2+ people from person blobs (no reliable faces).

- **Scoring**
  - Dominated by total people/faces count with bonuses when both detectors agree, plus sharpness.

- **Selection**
  - Uses the common tier-by-tier selection algorithm (tier → score → area).

---

### Mode: `bridge`

Goal: capture the walker on the skyline **near the end of the scan window**, preferring the best face quality, while strictly ignoring the waiting area (guide region).

- **Detection**
  - Person detection via background subtraction + face detection/mesh for tiering.
  - Time gating:
    - background-subtractor warmup (~0.7s)
    - only scans the **last 5 seconds** of the region
  - Guide region:
    - uses a **narrower** guide region width (25% of frame width)
    - anything inside the guide region is **always rejected** (even if it has a face)

- **Tiers**
  - **Tier 4**: face + frontal + smiling + eyes open
  - **Tier 3**: face + frontal
  - **Tier 2**: face (non-frontal) but valid size
  - **Tier 1**: person-only (no face)
  - **Tier 0**: obstruction (present in the general pipeline; filtered out before selection)

- **Scoring**
  - Same face scoring shape as `going` (confidence + sharpness + smile/eyes bonuses).
  - Person-only fallback: area + sharpness.

- **Selection**
  - Sort emphasizes **tier first**, then **closeness** (area), then score.
  - Additionally restricts candidates (per tier, when possible) to the **last ~1 second** of candidate times so the selected frame(s) tend to be from when the walker is closest.

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

- **Going mode**: Draws face boxes and shows tier/smile/eyes/sharpness; falls back to a person box when no face candidate is available.
- **Coming mode**: Draws guide region border, person boxes, face hints, obstruction labeling (Tier 0), and accept/reject reasons.
- **Group mode**: Draws people and faces plus counts and tier/score information.
- **Bridge mode**: Similar to coming overlays but with bridge-specific guide region and face-tier labeling; focuses on the tail window.
- Displays current detection mode, candidate counts, sharpness, confidence, and accept/reject reasons.
- Press `q` to stop processing early (the function returns the current error state).

---
