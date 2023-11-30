import cv2
from tqdm import tqdm

def get_crop_dimensions(crop_tops, crop_width, crop_height):
    return [(x, y, crop_width, crop_height) for x, y in crop_tops]

# Define your crop top-left corners and uniform width and height
crop_tops = [(235,45),(485,45),(735,45),(990,45),(1250,45),(1500,45),
             (235,295),(485,295),(735,295),(990,295),(1250,295),(1500,295),
             (235,550),(490,550),(745,550),(990,550),(1250,550),(1500,550),
             (235,805),(490,805),(745,805),(990,805),(1250,805),(1500,805)
]  # Add your coordinates here
crop_width, crop_height = 250, 250  # Set your desired width and height

# Initialize video capture and get video properties
video_path = '/home/frogtracker/P1030820.MP4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get crop dimensions
crops = get_crop_dimensions(crop_tops, crop_width, crop_height)

# Create VideoWriter objects for each crop
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writers = [
    cv2.VideoWriter(f'output_{i:0{2}d}.mp4', fourcc, fps, (crop_width, crop_height))
    for i, _ in enumerate(crops)
]


with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create and write cropped frames
        for i, (x, y, cw, ch) in enumerate(crops):
            cropped_frame = frame[y:y+ch, x:x+cw]
            writers[i].write(cropped_frame)

        pbar.update(1)  # Update progress bar per frame

# Release everything
cap.release()
for w in writers:
    w.release()