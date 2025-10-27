import cv2
import mediapipe as mp
import numpy as np
import os
from moviepy import VideoFileClip
from datetime import datetime

def visualize_pose_on_video(video_path, output_path="data/visualized_poses"):
    """
    Create a video with pose landmarks overlaid to verify pose extraction is working
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize MediaPipe pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer with unique timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(video_path).replace('.mp4', '')
    output_filename = os.path.join(output_path, f"{base_name}_pose_visualized_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    frame_count = 0
    poses_detected = 0
    
    print(f"🎬 Processing video: {video_path}")
    print(f"📐 Video dimensions: {width}x{height} @ {fps} FPS")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        
        # Draw pose landmarks if detected
        if result.pose_landmarks:
            poses_detected += 1
            mp_drawing.draw_landmarks(
                frame, 
                result.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Pose Detected: {poses_detected}/{frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # No pose detected
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "No Pose Detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"📊 Processed {frame_count} frames, poses detected: {poses_detected}")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"✅ Visualization complete!")
    print(f"📁 Output saved to: {output_filename}")
    print(f"📈 Summary: {poses_detected}/{frame_count} frames had poses detected ({poses_detected/frame_count*100:.1f}%)")
    
    return output_filename

def extract_and_visualize_poses(video_path):
    """
    Complete pipeline: extract poses and create visualization
    """
    from pose_extraction import extract_pose
    
    print("🔍 Step 1: Extracting pose keypoints...")
    pose_file, pose_count = extract_pose(video_path)
    
    if pose_file is None or pose_count == 0:
        print("❌ No poses detected in video.")
        return None, None
    
    print("🎨 Step 2: Creating pose visualization...")
    viz_file = visualize_pose_on_video(video_path)
    
    print("🎉 Complete! You now have:")
    print(f"   📊 Pose data: {pose_file}")
    print(f"   🎬 Visualization: {viz_file}")
    
    return pose_file, viz_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize pose extraction on video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--extract", action="store_true", help="Also extract pose keypoints")
    
    args = parser.parse_args()
    
    if args.extract:
        extract_and_visualize_poses(args.video)
    else:
        visualize_pose_on_video(args.video)
