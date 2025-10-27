import cv2
import json
import os
import argparse
from pathlib import Path

class VideoLabeler:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.labels_file = os.path.join(data_dir, "labels.json")
        self.labels = self.load_labels()
    
    def load_labels(self):
        """Load existing labels from JSON file"""
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_labels(self):
        """Save labels to JSON file"""
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
    
    def label_video(self, video_path, pose_file, label=None):
        """
        Display video and prompt for label
        label can be: 'left', 'right', 'center', or None (prompt user)
        """
        import threading
        import time
        
        # If label is provided directly, just save it
        if label:
            self.labels[pose_file] = {
                'label': label,
                'video': video_path
            }
            self.save_labels()
            return label
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Store current label
        current_label = "center"
        
        print("\n" + "="*50)
        print("VIDEO LABELING")
        print("="*50)
        print(f"Video: {video_path}")
        print(f"\n📺 A video window will open. Watch the video, then:")
        print("   Type 'L' for LEFT")
        print("   Type 'R' for RIGHT")
        print("   Type 'C' for CENTER")
        print("   Type 'Q' to quit without saving")
        print("="*50)
        
        def get_user_input():
            """Get input from terminal while video plays"""
            nonlocal current_label
            while True:
                try:
                    user_input = input("\nEnter label (L/R/C) or Q to quit: ").strip().lower()
                    if user_input in ['l', 'left']:
                        current_label = "left"
                        print(f"✅ Label set to: {current_label}")
                    elif user_input in ['r', 'right']:
                        current_label = "right"
                        print(f"✅ Label set to: {current_label}")
                    elif user_input in ['c', 'center']:
                        current_label = "center"
                        print(f"✅ Label set to: {current_label}")
                    elif user_input in ['q', 'quit']:
                        print("\n❌ Exited without saving")
                        return False
                    else:
                        print("Invalid input! Use L, R, C, or Q")
                except EOFError:
                    return False
                except Exception as e:
                    return False
        
        # Start input thread
        input_thread = threading.Thread(target=get_user_input, daemon=True)
        input_thread.start()
        
        # Play video in loop
        while cap.isOpened() and input_thread.is_alive():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Display label on frame
            height, width = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Current Label: {current_label.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Watch video, then type L/R/C in terminal", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press ESC or close window to save and continue", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            cv2.imshow('Video Labeler - Watch then type L/R/C in terminal', frame)
            
            # Check for ESC key or window close
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == 27:  # ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save the label
        self.labels[pose_file] = {
            'label': current_label,
            'video': video_path
        }
        self.save_labels()
        print(f"\n✅ Saved label: {current_label} for {pose_file}")
        
        return current_label
    
    def batch_label(self):
        """Label all unlabeled videos"""
        poses_dir = os.path.join(self.data_dir, "poses")
        clips_dir = os.path.join(self.data_dir, "clips")
        
        if not os.path.exists(poses_dir):
            print("❌ No poses directory found")
            return
        
        pose_files = [f for f in os.listdir(poses_dir) if f.endswith('.npy')]
        unlabeled = [f for f in pose_files if f not in self.labels]
        
        if not unlabeled:
            print("✅ All videos are already labeled!")
            return
        
        print(f"\n📊 Found {len(unlabeled)} unlabeled videos")
        for i, pose_file in enumerate(unlabeled, 1):
            print(f"\n[{i}/{len(unlabeled)}] Labeling: {pose_file}")
            
            # Find corresponding video (remove _pose_timestamp and change to .mp4)
            base_name = pose_file.rsplit('_pose_', 1)[0] + '.mp4'
            video_path = os.path.join(clips_dir, base_name)
            
            if not os.path.exists(video_path):
                print(f"⚠️  Video not found: {video_path}")
                continue
            
            label = self.label_video(video_path, pose_file)
            if label is None:
                print("Skipping remaining videos...")
                break
    
    def list_labels(self):
        """Display all current labels"""
        print("\n📋 Current Labels:")
        print("="*60)
        for pose_file, data in self.labels.items():
            print(f"{data['label']:8s} → {pose_file}")
        print(f"\nTotal: {len(self.labels)} labeled videos")
        print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label penalty kick videos")
    parser.add_argument("--video", type=str, help="Path to specific video to label")
    parser.add_argument("--pose", type=str, help="Path to corresponding pose file")
    parser.add_argument("--label", choices=['left', 'right', 'center'], help="Direct label")
    parser.add_argument("--batch", action="store_true", help="Label all unlabeled videos")
    parser.add_argument("--list", action="store_true", help="List all labels")
    
    args = parser.parse_args()
    
    labeler = VideoLabeler()
    
    if args.list:
        labeler.list_labels()
    elif args.batch:
        labeler.batch_label()
    elif args.video and args.pose:
        labeler.label_video(args.video, args.pose, args.label)
    else:
        # Interactive mode - show menu
        print("🏷️  Penalty Kick Video Labeler")
        print("\nOptions:")
        print("1. Label unlabeled videos (batch mode)")
        print("2. View current labels")
        print("\nRun with --batch or --list for quick access")
