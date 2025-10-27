import os
import yt_dlp
from moviepy import VideoFileClip
from datetime import datetime

def download_video(url: str, out_path="data/raw_videos", filename=None):
    os.makedirs(out_path, exist_ok=True)
    
    # Generate unique filename with timestamp if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"penalty_raw_{timestamp}"
    
    output_file = f"{out_path}/{filename}.mp4"
    ydl_opts = {
        "outtmpl": output_file,
        "format": "best[ext=mp4]/best",  # More flexible format selection
        "quiet": False,
        "retries": 3,  # Retry on failure
        "fragment_retries": 3,  # Retry individual fragments
        "noplaylist": True,  # Only download single video, not playlists
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"✅ Downloaded: {output_file}")
    return output_file


def trim_video(in_path, start, end, out_path="data/clips"):
    os.makedirs(out_path, exist_ok=True)
    clip = VideoFileClip(in_path)
    sub = clip.subclipped(start, end)
    
    # Generate unique filename with timestamp and time range
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(in_path).replace('.mp4', '')
    out_file = f"{out_path}/{base_name}_trimmed_{start}s-{end}s_{timestamp}.mp4"
    
    sub.write_videofile(out_file, codec="libx264", audio=False)
    print(f"✂️ Trimmed clip saved at: {out_file}")
    return out_file


if __name__ == "__main__":
    url = input("YouTube URL: ")
    start = float(input("Start time (s): "))
    end = float(input("End time (s): "))

    raw = download_video(url)  # Will auto-generate unique filename
    trim_video(raw, start, end)
