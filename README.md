# Penalty Kick Direction Predictor

Predicts penalty kick direction (Left / Center / Right) from pre-shot player motion using pose estimation with MediaPipe and machine learning.

## Overview

This project extracts human pose keypoints from penalty kick videos and uses them to predict shot direction. The system includes:

- **Video processing pipeline**: Download, trim, extract poses, visualize
- **Interactive trimming tool**: Precise clip extraction with visual preview
- **Batch processing**: Process multiple clips efficiently
- **Pose detection**: MediaPipe-based pose extraction with normalization
- **Labeling system**: Interactive labeling workflow
- **Visualization**: Preview poses overlaid on video

## Setup

### Prerequisites
- Python 3.8+
- ffmpeg (for video processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourname/penaltyPredictor
cd penaltyPredictor

# Create virtual environment
python -m venv venv_penalty
source venv_penalty/bin/activate  # On Windows: venv_penalty\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `yt-dlp` - YouTube video downloading
- `mediapipe` - Pose detection
- `opencv-python` - Video processing
- `moviepy` - Video editing and preview
- `numpy` - Data handling

## Quick Start

### 1. Collect Data with Interactive Trim (Recommended)

The easiest way to collect data is using the interactive trim tool:

```bash
# Download and process a video with multiple penalty kicks
python src/interactive_trim.py --url "https://youtube.com/watch?v=..."
```

This will:
1. Download the full video
2. Let you navigate and precisely mark START and END points
3. Preview the clip before saving
4. Automatically extract poses
5. Create visualization
6. Let you label it (LEFT, CENTER, or RIGHT)
7. Ask if you want to process another clip from the same video

### 2. Use Existing Video

If you already have a video file:

```bash
python src/interactive_trim.py --video "path/to/video.mp4"
```

## Interactive Trim Tool

### Features
- **Visual timeline**: See exactly where you are in the video
- **Frame-by-frame navigation**: Precise 0.1s steps
- **Preview before saving**: Watch your clip before processing
- **Auto-continue**: Process multiple clips in one session
- **Smart skipping**: Automatically skips clips with no detected poses

### Controls
```
A / D or ← / →  - Navigate backward/forward (0.1s steps)
Q               - Mark START point
E               - Mark END point
S               - Preview trim (watch the clip)
W               - Save and process (extract poses, visualize, label)
X               - Exit (with confirmation)
```

### Workflow

1. **Download video** - Specify YouTube URL or existing file
2. **Navigate** - Use A/D to find the start of the penalty kick
3. **Mark START (Q)** - When the run-up begins
4. **Navigate** - Find the moment just after contact with ball
5. **Mark END (E)** - After kick but before seeing ball direction
6. **Preview (S)** - Watch the clip (optional)
7. **Save (W)** - Process the clip (extract poses, visualize, label)
8. **Label** - Type L (LEFT), R (RIGHT), or C (CENTER) in terminal
9. **Continue?** - Process another clip or finish

### Trimming Guidelines

**What to Capture:**
- ✅ 1-2 seconds before ball contact (run-up, approach)
- ✅ The moment of kick contact
- ✅ Just after contact (but before seeing direction)

**What to Avoid:**
- ❌ Ball trajectory and direction (defeats prediction purpose)
- ❌ Goal celebration/outcome
- ❌ Only run-up without contact

**Duration**: Typically 2-3 seconds total

### Example Session

```bash
$ python src/interactive_trim.py --url "https://youtube.com/watch?v=abc123"

🎬 Interactive Video Trimming Tool
==================================================
📥 Step 1: Downloading video from https://youtube.com/watch?v=abc123
[Downloads...]

📹 Video Info:
   Duration: 125.30 seconds
   FPS: 30
   Total frames: 3759

🎮 Controls:
   A/D or ←/→ - Navigate backward/forward (0.1s steps)
   Q - Mark START point
   E - Mark END point
   S - Preview trim (watch the clip before saving)
   W - Save and process (extract poses, visualize, label)
   X - Exit (will confirm first)

[Navigate with A/D, mark Q and E]

📹 Clip #1
✅ Saved: penalty_raw_trimmed_10.5s-12.8s.mp4
🔍 Extracting pose keypoints...
✅ Pose saved at: data/poses/penalty_raw_trimmed_10.5s-12.8s_pose.npy (60 frames with poses)
🎨 Creating pose visualization...
🏷️ Label this clip
[Shows video, type 'R' in terminal]
✅ Labeled as: right

Trim and label another clip from this video? (y/n): y

📹 Clip #2
[... repeat ...]
```

## Data Structure

```
penaltyPredictor/
├── src/
│   ├── interactive_trim.py    # Interactive trimming & labeling (RECOMMENDED)
│   ├── batch_process.py       # Batch process multiple clips
│   ├── pipeline.py            # Full pipeline (download + process)
│   ├── data_collection.py     # Video download utilities
│   ├── pose_extraction.py     # MediaPipe pose extraction
│   ├── pose_visualization.py  # Create pose overlay videos
│   └── labeler.py             # Labeling system
├── data/
│   ├── raw_videos/           # Full downloaded videos
│   ├── clips/                # Trimmed penalty clips
│   ├── poses/                # Pose keypoint files (.npy)
│   ├── visualized_poses/     # Videos with pose overlays
│   └── labels.json          # Labels for each clip
└── models/                   # (Future) Trained models
```

### Pose Data Format

Each `.npy` file contains:
- **Shape**: `(frames, 33, 2)`
- **Frames**: Number of video frames with detected poses
- **33 keypoints**: MediaPipe pose landmarks
- **2 coordinates**: Normalized (x, y) positions
  - Normalized by hip center (position) and shoulder width (scale)

### Labels Format

`data/labels.json`:
```json
{
  "penalty_trimmed_10.5s-12.8s_pose_20250127_123456.npy": {
    "label": "right",
    "video": "data/clips/penalty_trimmed_10.5s-12.8s.mp4"
  },
  ...
}
```

## Other Tools

### Batch Process Multiple Clips

Process several clips from a long video at once:

```bash
python src/batch_process.py --url "URL" --clips 10:12 45:47 120:122

# Or use existing video
python src/batch_process.py --use-existing --clips 10.5:12.8 45.2:47.5
```

### Interactive Batch Mode

Process clips one-by-one with prompts:

```bash
python src/batch_process.py --url "URL" --interactive

# For each clip:
Start time (seconds, decimals OK): 10.5
End time (seconds, decimals OK): 12.3
[Processes and labels]
Process another clip? (y/n): y
```

### Full Pipeline (Single Clip)

Process one clip completely:

```bash
python src/pipeline.py --url "URL" --start 10.5 --end 12.8
```

### Label Existing Data

Label videos that have been processed but not labeled:

```bash
# Label all unlabeled clips
python src/labeler.py --batch

# List current labels
python src/labeler.py --list

# Label specific video
python src/labeler.py --video data/clips/video.mp4 --pose data/poses/video_pose.npy
```

## Development Status

### ✅ Completed
- Video download from YouTube
- Precise video trimming (frame-accurate)
- Interactive preview and trimming tool
- Pose detection with MediaPipe
- Pose normalization for scale/position invariance
- Automatic visualization generation
- Interactive labeling system
- Batch processing capabilities
- Skip videos with no detected poses
- Decimal-second precision timing

### 🔨 In Progress
- Pose feature extraction from keypoints
- Model architecture design
- Training data collection

### 📋 TODO
- Implement model (LSTM/Transformer for sequences)
- Train on labeled data
- Evaluation metrics
- Inference on new videos
- Deployment

## Tips

1. **Collect diverse data**: Mix of players, camera angles, kicking styles
2. **Accurate trimming**: Don't include ball flight - defeats the purpose
3. **Label consistently**: Left = goalkeeper's left (from shooter's perspective)
4. **Handle errors**: Tool auto-skips clips with no poses detected
5. **Use previews**: Always preview (press S) before saving to avoid bad clips

## Troubleshooting

**No poses detected?**
- Player might be too small or occluded
- Camera angle might be poor
- Motion blur affects detection
- Solution: Skip and try another clip

**Video download fails?**
- YouTube might have restrictions
- Try different video or check yt-dlp version: `pip install --upgrade yt-dlp`

**Labeling window not responding?**
- Make sure to click on the OpenCV window
- Type labels in the terminal, not video window

## License

MIT License

## Contributors

Your Name - https://github.com/yourname

## Acknowledgments

- MediaPipe for pose detection
- MoviePy for video processing
- yt-dlp for YouTube downloads
