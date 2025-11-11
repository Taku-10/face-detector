
## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Install Dependencies

1. Navigate to the project directory:
   ```bash
   cd face-detector
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Program

1. **Edit the video path in `face-detection.py`**:
   
   Open `face-detection.py` and modify the `input_video_path` parameter around line 402:
   
   ```python
   result = detect_zipline_segment(
       input_video_path="your-video-file.MP4",  # Change this to your video path
       direction="going",  # or "coming"
       min_duration=2.0,
       max_duration=20.0,  # Optional: set to None for no limit
   )
   ```

2. **Run the script**:
   ```bash
   python3 face-detection.py
   ```

### Parameters

- `input_video_path` (string): Path to your video file (e.g., `"GH011700.MP4"` or `"/path/to/video.MP4"`)
- `direction` (string): Either `"coming"` or `"going"`
  - `"coming"`: Uses motion detection to find when rider approaches
  - `"going"`: Uses face detection to find when rider looks at camera
- `min_duration` (float): Minimum duration in seconds (default: 2.0)
- `max_duration` (float, optional): Maximum duration cap in seconds (default: 20.0, set to `None` for no limit)

### Output

The program outputs a JSON object with the following fields:

```json
{
  "input_video": "your-video-file.MP4",
  "direction": "going",
  "start_time": 2.15,
  "end_time": 7.50,
  "duration": 5.35,
  "valid": true
}
```

- `start_time`: When the rider appears/looks at camera (seconds)
- `end_time`: When the rider leaves or end of video (seconds)
- `duration`: Total duration of the segment (seconds)
- `valid`: Whether detection was successful
- `reason`: Error message if `valid` is `false`

### Example

```python
result = detect_zipline_segment(
    input_video_path="GH011700.MP4",
    direction="going",
    min_duration=2.0,
    max_duration=None,  # No limit
)
```
