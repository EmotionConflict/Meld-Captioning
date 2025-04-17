import cv2

def extract_middle_frame(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(save_path, frame)
    cap.release()
