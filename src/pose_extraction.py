import cv2, mediapipe as mp, numpy as np, os, argparse

#Normalize pose for different scalings
def normalize_pose(pose):
    # pose shape: (33, 2)
    left_hip, right_hip = pose[23], pose[24]
    center_hip = (np.array(left_hip) + np.array(right_hip)) / 2
    left_shoulder, right_shoulder = pose[11], pose[12]
    scale = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder)) + 1e-6
    return (pose - center_hip) / scale


def extract_pose(video_path, save_dir="data/poses"):
    os.makedirs(save_dir, exist_ok=True)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        if result.pose_landmarks:
            kp = [(lm.x, lm.y) for lm in result.pose_landmarks.landmark]
            kp = normalize_pose(kp)
            keypoints.append(kp)

    out_file = os.path.join(save_dir, os.path.basename(video_path).replace(".mp4", "_pose.npy"))
    np.save(out_file, np.array(keypoints))
    print(f"✅ Pose saved at: {out_file}")
    return out_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()
    extract_pose(args.video)