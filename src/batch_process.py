#!/usr/bin/env python3
"""
Batch Process Penalty Clips
============================

Download one video with multiple penalties, then process clips one by one.

Usage:
    python batch_process.py --url "URL" --clips START1:END1 START2:END2 ...
    
Example:
    python batch_process.py --url "https://youtube.com/watch?v=abc" --clips 10:12 45:47 120:122
    
    # With decimal seconds for precise timing:
    python batch_process.py --url "URL" --clips 10.5:12.3 45.2:47.8
"""

import os
import sys
import argparse
from data_collection import download_video, trim_video
from pose_extraction import extract_pose
from pose_visualization import visualize_pose_on_video
from labeler import VideoLabeler

def parse_clip_range(clip_str):
    """Parse '10:12' or '10.5:12.3' into (10.0, 12.0) or (10.5, 12.3)
    Supports decimal seconds for precise trimming
    """
    try:
        start, end = clip_str.split(':')
        return float(start), float(end)
    except:
        raise ValueError(f"Invalid clip format: {clip_str}. Use START:END (e.g., 10:12 or 10.5:12.3)")

def batch_process_clips(url, clip_ranges, output_dir="data", download_new=True):
    """
    Download video once, then process multiple clips from it
    
    Args:
        url: YouTube URL
        clip_ranges: List of (start, end) tuples
        output_dir: Where to save files
        download_new: Whether to download or use existing video
    """
    
    # Step 1: Download the long video (only once)
    if download_new:
        print("🚀 Starting Batch Penalty Predictor Pipeline")
        print("=" * 50)
        print(f"📥 Step 1: Downloading video with multiple penalties from {url}")
        raw_video = download_video(url, out_path=f"{output_dir}/raw_videos")
        print(f"✅ Full video downloaded: {raw_video}\n")
    else:
        # Find the most recent raw video
        raw_videos = os.listdir(f"{output_dir}/raw_videos")
        if not raw_videos:
            print("❌ No raw videos found! Run with download_new=True first.")
            return
        
        # Get the most recent one
        raw_videos.sort(reverse=True)
        raw_video = os.path.join(f"{output_dir}/raw_videos", raw_videos[0])
        print(f"📁 Using existing video: {raw_video}\n")
    
    # Step 2: Process each clip
    labeler = VideoLabeler(output_dir)
    results = []
    
    for i, (start, end) in enumerate(clip_ranges, 1):
        print("\n" + "="*50)
        print(f"📹 Processing Clip {i}/{len(clip_ranges)}: {start}s - {end}s")
        print("="*50)
        
        # Trim video
        print(f"✂️ Trimming video...")
        trimmed_video = trim_video(raw_video, start, end, out_path=f"{output_dir}/clips")
        
        # Extract poses
        print("🔍 Extracting pose keypoints...")
        pose_file, pose_count = extract_pose(trimmed_video, save_dir=f"{output_dir}/poses")
        
        # Skip if no poses detected
        if pose_file is None or pose_count == 0:
            print(f"⚠️  Skipping clip {i}: No poses detected")
            continue
        
        # Create visualization
        print("🎨 Creating pose visualization...")
        viz_file = visualize_pose_on_video(trimmed_video, output_path=f"{output_dir}/visualized_poses")
        
        # Label the clip
        print(f"\n🏷️ Label Clip {i} (LEFT, CENTER, or RIGHT)")
        label = labeler.label_video(trimmed_video, pose_file)
        
        if label:
            print(f"✅ Clip {i} labeled as: {label}")
            results.append({
                'clip': i,
                'time_range': f"{start}s-{end}s",
                'label': label,
                'video': trimmed_video,
                'pose': pose_file,
                'viz': viz_file
            })
        else:
            print(f"⚠️ Clip {i} was not labeled")
    
    # Summary
    print("\n" + "="*50)
    print("🎉 Batch Processing Complete!")
    print("="*50)
    print(f"📊 Processed {len(results)} clips")
    print("\n📋 Summary:")
    for r in results:
        print(f"   Clip {r['clip']} ({r['time_range']}): {r['label']}")
    print("="*50)
    
    return results

def interactive_mode(url, output_dir="data", download_new=True):
    """Interactive mode: download video, then process clips one by one"""
    
    # Download the full video
    if download_new:
        print("🚀 Starting Interactive Penalty Predictor")
        print("=" * 50)
        print(f"📥 Step 1: Downloading video from {url}")
        raw_video = download_video(url, out_path=f"{output_dir}/raw_videos")
    else:
        raw_videos = os.listdir(f"{output_dir}/raw_videos")
        if not raw_videos:
            print("❌ No raw videos found!")
            return
        raw_videos.sort(reverse=True)
        raw_video = os.path.join(f"{output_dir}/raw_videos", raw_videos[0])
        print(f"📁 Using existing video: {raw_video}")
    
    print(f"\n✅ Full video ready: {raw_video}")
    print("\nNow you can process clips from this video.")
    print("Press Ctrl+C to exit or just keep processing clips.")
    
    labeler = VideoLabeler(output_dir)
    clip_count = 0
    
    try:
        while True:
            print("\n" + "="*50)
            clip_count += 1
            print(f"📹 Clip #{clip_count}")
            print("="*50)
            
            start = float(input("Start time (seconds, decimals OK): "))
            end = float(input("End time (seconds, decimals OK): "))
            
            # Trim video
            print(f"✂️ Trimming {start}s - {end}s...")
            trimmed_video = trim_video(raw_video, start, end, out_path=f"{output_dir}/clips")
            
            # Extract poses
            print("🔍 Extracting pose keypoints...")
            pose_file, pose_count = extract_pose(trimmed_video, save_dir=f"{output_dir}/poses")
            
            # Skip if no poses detected
            if pose_file is None or pose_count == 0:
                print("⚠️  Skipping: No poses detected in this clip")
                continue
            
            # Create visualization
            print("🎨 Creating pose visualization...")
            viz_file = visualize_pose_on_video(trimmed_video, output_path=f"{output_dir}/visualized_poses")
            
            # Label
            print("\n🏷️ Label this clip")
            label = labeler.label_video(trimmed_video, pose_file)
            
            if label:
                print(f"✅ Labeled as: {label}")
            else:
                print("⚠️ Not labeled")
            
            # Ask if continue
            continue_processing = input("\nProcess another clip? (y/n): ").strip().lower()
            if continue_processing != 'y':
                print("\n👋 Done!")
                break
                
    except KeyboardInterrupt:
        print("\n\n👋 Exited by user")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process penalty kick clips")
    
    parser.add_argument("--url", type=str, help="YouTube URL to download")
    parser.add_argument("--clips", nargs='+', help="Clip ranges (e.g., 10:12 45:47 or 10.5:12.3 for precise timing)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode (process clips one by one)")
    parser.add_argument("--use-existing", action="store_true", help="Use existing video instead of downloading")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    if args.interactive:
        if not args.url and not args.use_existing:
            print("❌ Need --url or --use-existing for interactive mode")
            sys.exit(1)
        interactive_mode(args.url, args.output, download_new=not args.use_existing)
    elif args.clips:
        # Batch mode
        if not args.url and not args.use_existing:
            print("❌ Need --url or --use-existing")
            sys.exit(1)
        
        clip_ranges = [parse_clip_range(c) for c in args.clips]
        batch_process_clips(args.url, clip_ranges, args.output, download_new=not args.use_existing)
    else:
        print("❌ Need --clips or --interactive")
        print("\nUsage examples:")
        print("  Batch mode:    python batch_process.py --url URL --clips 10:12 45:47")
        print("  Interactive:   python batch_process.py --url URL --interactive")
        print("  Use existing:  python batch_process.py --use-existing --interactive")
        sys.exit(1)
