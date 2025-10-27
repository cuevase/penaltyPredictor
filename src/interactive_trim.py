#!/usr/bin/env python3
"""
Interactive Video Trimming Tool
=================================

Download video from YouTube, preview it, trim multiple clips with precise control, 
extract poses, and label each one.

Features:
- Visual timeline with exact frame navigation
- Preview before saving
- Continuously process multiple clips from same video
- Extract poses and create visualizations
- Label each clip as LEFT, CENTER, or RIGHT

Usage:
    python interactive_trim.py --url "URL"
    python interactive_trim.py --video "path/to/video.mp4"  # Use existing video

Controls:
- A/D or ←/→: Navigate frame-by-frame
- Q: Mark START point
- E: Mark END point  
- S: Preview trim before saving
- W: Save and process clip (extract poses, visualize, label)
- X: Exit (confirms first)
"""

import cv2
import os
import sys
import argparse
from moviepy import VideoFileClip
from data_collection import download_video
from pose_extraction import extract_pose
from pose_visualization import visualize_pose_on_video
from labeler import VideoLabeler

class VideoPreviewer:
    """Interactive video player for finding exact trim points"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        # Current position
        self.current_frame = 0
        
        # Trim points
        self.start_time = 0.0
        self.end_time = self.duration
        
        print(f"\n📹 Video Info:")
        print(f"   Duration: {self.duration:.2f} seconds")
        print(f"   FPS: {self.fps}")
        print(f"   Total frames: {self.total_frames}")
        print(f"\n🎮 Controls:")
        print(f"   A/D or ←/→ - Navigate backward/forward (0.1s steps)")
        print(f"   Q - Mark START point")
        print(f"   E - Mark END point")
        print(f"   S - Preview trim (watch the clip before saving)")
        print(f"   W - Save and process (extract poses, visualize, label)")
        print(f"   X - Exit (will confirm first)")
    
    def get_frame(self, frame_num):
        """Get a specific frame"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        return ret, frame
    
    def play_around_position(self):
        """Play video around current position"""
        ret, frame = self.get_frame(self.current_frame)
        if not ret:
            return None
        
        # Add info overlay
        height, width = frame.shape[:2]
        current_time = self.current_frame / self.fps
        
        # Draw timeline
        timeline_width = width - 40
        timeline_x = 20
        
        # Draw timeline background
        cv2.rectangle(frame, (timeline_x, height - 80), (timeline_x + timeline_width, height - 60), (100, 100, 100), -1)
        
        # Current position marker
        current_pos = (current_time / self.duration) * timeline_width
        cv2.circle(frame, (timeline_x + int(current_pos), height - 70), 5, (0, 255, 0), -1)
        
        # Draw start/end markers
        if self.start_time > 0:
            start_pos = (self.start_time / self.duration) * timeline_width
            cv2.rectangle(frame, (timeline_x + int(start_pos) - 2, height - 90), 
                        (timeline_x + int(start_pos) + 2, height - 60), (255, 0, 0), -1)
            cv2.putText(frame, "START", (timeline_x + int(start_pos) - 20, height - 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        if self.end_time < self.duration:
            end_pos = (self.end_time / self.duration) * timeline_width
            cv2.rectangle(frame, (timeline_x + int(end_pos) - 2, height - 90), 
                        (timeline_x + int(end_pos) + 2, height - 60), (0, 0, 255), -1)
            cv2.putText(frame, "END", (timeline_x + int(end_pos) - 10, height - 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Add text info
        text_y = 30
        cv2.rectangle(frame, (0, 0), (width, 150), (0, 0, 0), -1)
        cv2.putText(frame, f"Time: {current_time:.2f}s", (10, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Start: {self.start_time:.2f}s", (10, text_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"End: {self.end_time:.2f}s", (10, text_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Trim Duration: {self.end_time - self.start_time:.2f}s", (10, text_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, "A/D: Backward/Forward | Q: Mark Start | E: Mark End | S: Preview | W: Save | X: Exit", 
                   (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow('Video Previewer - Navigate and trim precisely', frame)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('x'):
            return 'cancel'
        elif key == ord('a') or key == 81:  # A or left arrow
            self.current_frame = max(0, self.current_frame - int(self.fps * 0.1))  # Go back 0.1 sec
        elif key == ord('d') or key == 83:  # D or right arrow
            self.current_frame = min(self.total_frames - 1, self.current_frame + int(self.fps * 0.1))  # Forward 0.1 sec
        elif key == ord('q'):
            # Mark start
            self.start_time = current_time
            return 'start_marked'
        elif key == ord('e'):
            # Mark end
            self.end_time = current_time
            return 'end_marked'
        elif key == ord('s'):
            # Show preview
            if self.end_time > self.start_time:
                return 'preview'
            else:
                print("⚠️  Mark start (Q) and end (E) first!")
        elif key == ord('w'):
            # Confirm and save
            if self.end_time <= self.start_time:
                print("⚠️  Start time must be before end time!")
                print(f"   Current: Start={self.start_time:.2f}s, End={self.end_time:.2f}s")
            else:
                return 'save'
        
        return None
    
    def preview_trim(self, output_dir="data/clips"):
        """Preview the trimmed clip"""
        print(f"\n🎬 Previewing trim: {self.start_time:.2f}s - {self.end_time:.2f}s")
        
        clip = VideoFileClip(self.video_path)
        preview_clip = clip.subclipped(self.start_time, self.end_time)
        
        # Play preview
        preview_clip.preview()
        preview_clip.close()
        
        # Ask if user wants to save
        response = input("\n💾 Save this trim? (y/n): ").strip().lower()
        if response == 'y':
            return True
        return False
    
    def save_trim(self, output_dir="data/clips"):
        """Save the trimmed clip"""
        os.makedirs(output_dir, exist_ok=True)
        
        clip = VideoFileClip(self.video_path)
        trimmed = clip.subclipped(self.start_time, self.end_time)
        
        # Generate filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(self.video_path).replace('.mp4', '')
        out_file = f"{output_dir}/{base_name}_trimmed_{self.start_time:.2f}s-{self.end_time:.2f}s_{timestamp}.mp4"
        
        trimmed.write_videofile(out_file, codec="libx264", audio=False)
        trimmed.close()
        clip.close()
        
        print(f"✅ Saved: {out_file}")
        return out_file
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()

def interactive_trim(url=None, video_path=None, output_dir="data"):
    """Main interactive trimming workflow"""
    
    print("🎬 Interactive Video Trimming Tool")
    print("=" * 50)
    
    # Step 1: Download or use video
    if url:
        print(f"📥 Step 1: Downloading video from {url}")
        raw_video = download_video(url, out_path=f"{output_dir}/raw_videos")
    elif video_path:
        raw_video = video_path
        print(f"📁 Using existing video: {raw_video}")
    else:
        print("❌ Need --url or --video")
        return
    
    previewer = VideoPreviewer(raw_video)
    labeler = VideoLabeler(output_dir)
    clip_count = 0
    
    try:
        while True:
            # Reset trim points for new clip
            previewer.start_time = 0.0
            previewer.end_time = previewer.duration
            
            clip_count += 1
            print("\n" + "="*50)
            print(f"📹 Clip #{clip_count}")
            print("="*50)
            
            # Navigate and mark trim points
            while True:
                action = previewer.play_around_position()
                
                if action == 'cancel':
                    print("\n❌ Cancelled")
                    print("Are you sure you want to exit? (y/n): ", end='')
                    confirm = input().strip().lower()
                    if confirm == 'y':
                        previewer.cleanup()
                        return
                    continue
                elif action == 'start_marked':
                    print(f"✅ Start marked at {previewer.start_time:.2f}s")
                    continue
                elif action == 'end_marked':
                    print(f"✅ End marked at {previewer.end_time:.2f}s")
                    continue
                elif action == 'preview':
                    # Preview the trim
                    save = previewer.preview_trim(output_dir=f"{output_dir}/clips")
                    if save:
                        break
                elif action == 'save':
                    break
                elif action is None:
                    # Just a frame advance, continue
                    continue
            
            # Save the trimmed clip
            trimmed_video = previewer.save_trim(output_dir=f"{output_dir}/clips")
            
            # Extract poses
            print("\n🔍 Extracting pose keypoints...")
            pose_file, pose_count = extract_pose(trimmed_video, save_dir=f"{output_dir}/poses")
            
            if pose_file is None or pose_count == 0:
                print("⚠️  No poses detected. Skipping labeling.")
            else:
                # Create visualization
                print("🎨 Creating pose visualization...")
                viz_file = visualize_pose_on_video(trimmed_video, output_path=f"{output_dir}/visualized_poses")
                
                # Label
                print("\n🏷️ Label this clip")
                label = labeler.label_video(trimmed_video, pose_file)
                
                if label:
                    print(f"✅ Labeled as: {label}")
            
            # Ask if want to process another clip
            print("\n" + "="*50)
            continue_trimming = input("Trim and label another clip from this video? (y/n): ").strip().lower()
            if continue_trimming != 'y':
                print("\n👋 Done processing this video!")
                break
            
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    
    previewer.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive video trimming with preview")
    parser.add_argument("--url", type=str, help="YouTube URL to download")
    parser.add_argument("--video", type=str, help="Path to existing video")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    if not args.url and not args.video:
        print("❌ Need --url or --video")
        print("\nExamples:")
        print("  python interactive_trim.py --url 'https://youtube.com/watch?v=...'")
        print("  python interactive_trim.py --video 'data/raw_videos/video.mp4'")
        sys.exit(1)
    
    interactive_trim(args.url, args.video, args.output)
