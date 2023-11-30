from video_splitter import VideoSplitter as vs
import os



def process_folder(main_directory, path_for_output, frame_limit=float('inf')):
    first_iteration=True
    for root, dirs, files in os.walk(main_directory):
        # Exclude the "split_videos" directory
        if "split_videos" in dirs:
            dirs.remove("split_videos")
            continue
        
        for file in files:
            if file.lower().endswith(".mp4"):
                mp4_path = os.path.join(root, file)
                if first_iteration: 
                    splitter = vs(mp4_path, path_for_output,frame_limit=frame_limit)
                    first_iteration=False
                    print(f"Processing {mp4_path}")
                    splitter()
                else:
                    splitter.change_file(mp4_path)
                    print(f"Processing {mp4_path}")
                    splitter()


# Linux Filepaths
#directory=r'/media/frogtracker/Beck 07/ND250fpsOct23'
#path_for_output = r'/home/frogtracker/Videos/split_videos'


#windows Filepaths
directory = r'd:\ND250fpsOct23'
path_for_output = r'd:\ND250fpsOct23\split_videos'


frame_limit=10
process_folder(directory,path_for_output,frame_limit)