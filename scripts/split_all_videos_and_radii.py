from video_splitter import VideoSplitter as vs
import os
import csv


def process_folder(main_directory, path_for_output, frame_limit=float('inf')):
    first_iteration=True
    well_radii = []
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
                    well_rad = splitter.get_well_radius()
                    well_radii.append(well_rad)
                else:
                    splitter.change_file(mp4_path)
                    print(f"Processing {mp4_path}")
                    splitter()
                    well_rad = splitter.get_well_radius()
                    well_radii.append(well_rad)
    return well_radii


# Linux Filepaths
#directory=r'/media/frogtracker/Beck 07/ND250fpsOct23'
#path_for_output = r'/home/frogtracker/Videos/split_videos'


#windows Filepaths
directory = r'd:\ND250fpsOct23'
path_for_output = r'd:\ND250fpsOct23\split_videos'
csv_file_path = r'd:\ND250fpsOct23\well_sizes'



frame_limit=10
well_radii=process_folder(directory,path_for_output,frame_limit)

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Well Radii'])  # Writing header
    for radius in well_radii:
        writer.writerow([radius])  