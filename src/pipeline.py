#!/usr/bin/env python3
"""
Complete Penalty Predictor Pipeline
====================================

This script provides a complete workflow for:
1. Downloading videos from YouTube
2. Trimming videos to specific time ranges
3. Extracting pose keypoints
4. Visualizing pose extraction results

Usage:
    python pipeline.py --url "https://youtube.com/watch?v=..." --start 10 --end 20
"""

import os
import sys
import argparse
from data_collection import download_video, trim_video
from pose_extraction import extract_pose
from pose_visualization import visualize_pose_on_video
from labeler import VideoLabeler

def complete_pipeline(url, start_time, end_time, output_dir="data"):
    """
    Complete pipeline: download → trim → extract poses → visualize
    """
    print("🚀 Starting Penalty Predictor Pipeline")
    print("=" * 50)
    
    # Step 1: Download video
    print(f"📥 Step 1: Downloading video from {url}")
    raw_video = download_video(url, out_path=f"{output_dir}/raw_videos")  # Auto-generates unique name
    
    # Step 2: Trim video
    print(f"✂️ Step 2: Trimming video from {start_time}s to {end_time}s")
    trimmed_video = trim_video(raw_video, start_time, end_time, out_path=f"{output_dir}/clips")
    
    # Step 3: Extract poses
    print("🔍 Step 3: Extracting pose keypoints...")
    pose_file, pose_count = extract_pose(trimmed_video, save_dir=f"{output_dir}/poses")
    
    # Check if poses were detected
    if pose_file is None or pose_count == 0:
        print("⚠️  No poses detected in video. Skipping labeling.")
        print("\n🎉 Pipeline Complete!")
        print("=" * 50)
        print("📁 Generated files:")
        print(f"   📹 Raw video: {raw_video}")
        print(f"   ✂️ Trimmed video: {trimmed_video}")
        return {
            'raw_video': raw_video,
            'trimmed_video': trimmed_video,
            'pose_file': None,
            'visualization': None
        }
    
    # Step 4: Create visualization
    print("🎨 Step 4: Creating pose visualization...")
    viz_file = visualize_pose_on_video(trimmed_video, output_path=f"{output_dir}/visualized_poses")
    print("🏷️ Step 5: Label this clip (LEFT, CENTER, or RIGHT)")
    labeler = VideoLabeler(output_dir)
    label = labeler.label_video(trimmed_video, pose_file)
    
    if label:
        print(f"✅ Labeled as: {label}")
    print("\n🎉 Pipeline Complete!")
    print("=" * 50)
    print("📁 Generated files:")
    print(f"   📹 Raw video: {raw_video}")
    print(f"   ✂️ Trimmed video: {trimmed_video}")
    print(f"   📊 Pose data: {pose_file}")
    print(f"   🎬 Visualization: {viz_file}")
    
    return {
        'raw_video': raw_video,
        'trimmed_video': trimmed_video,
        'pose_file': pose_file,
        'visualization': viz_file
    }

def process_existing_video(video_path):
    """
    Process an existing video file (skip download step)
    """
    print(f"🎬 Processing existing video: {video_path}")
    print("=" * 50)
    
    # Extract poses
    print("🔍 Step 1: Extracting pose keypoints...")
    pose_file, pose_count = extract_pose(video_path, save_dir="data/poses")
    
    # Check if poses were detected
    if pose_file is None or pose_count == 0:
        print("⚠️  No poses detected in video.")
        return {
            'pose_file': None,
            'visualization': None
        }
    
    # Create visualization
    print("🎨 Step 2: Creating pose visualization...")
    viz_file = visualize_pose_on_video(video_path, output_path="data/visualized_poses")
    
    print("\n🎉 Processing Complete!")
    print("=" * 50)
    print("📁 Generated files:")
    print(f"   📊 Pose data: {pose_file}")
    print(f"   🎬 Visualization: {viz_file}")
    
    return {
        'pose_file': pose_file,
        'visualization': viz_file
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Penalty Predictor Pipeline")
    
    # Mode selection
    parser.add_argument("--mode", choices=["full", "existing"], default="full",
                       help="Pipeline mode: 'full' for download+process, 'existing' for process only")
    
    # Full pipeline arguments
    parser.add_argument("--url", type=str, help="YouTube URL to download")
    parser.add_argument("--start", type=float, help="Start time in seconds (decimals OK)")
    parser.add_argument("--end", type=float, help="End time in seconds (decimals OK)")
    
    # Existing video argument
    parser.add_argument("--video", type=str, help="Path to existing video file")
    
    # Output directory
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        if not all([args.url, args.start is not None, args.end is not None]):
            print("❌ Error: Full pipeline requires --url, --start, and --end arguments")
            sys.exit(1)
        
        complete_pipeline(args.url, args.start, args.end, args.output)
        
    elif args.mode == "existing":
        if not args.video:
            print("❌ Error: Existing video mode requires --video argument")
            sys.exit(1)
        
        if not os.path.exists(args.video):
            print(f"❌ Error: Video file not found: {args.video}")
            sys.exit(1)
        
        process_existing_video(args.video)
