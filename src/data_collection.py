import os
import yt_dlp
from moviepy import VideoFileClip

def download_video(url: str, out_path="data/raw_videos", filename="sample"):
    os.makedirs(out_path, exist_ok=True)
    output_file = f"{out_path}/{filename}.mp4"
    ydl_opts = {
        "outtmpl": output_file,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "quiet": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"✅ Downloaded: {output_file}")
    return output_file


def trim_video(in_path, start, end, out_path="data/clips"):
    os.makedirs(out_path, exist_ok=True)
    clip = VideoFileClip(in_path)
    sub = clip.subclip(start, end)
    out_file = f"{out_path}/{os.path.basename(in_path).replace('.mp4','')}_trimmed.mp4"
    sub.write_videofile(out_file, codec="libx264", audio=False)
    print(f"✂️ Trimmed clip saved at: {out_file}")
    return out_file


if __name__ == "__main__":
    url = input("YouTube URL: ")
    start = float(input("Start time (s): "))
    end = float(input("End time (s): "))

    raw = download_video(url, filename="penalty_raw")
    trim_video(raw, start, end)
