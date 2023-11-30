from video_splitter_for_stab import VideoSplitter as vs
import os




# Windows Filepaths
path_for_output = r'd:\ND250fpsOct23\stabilised_first_then_split'
path_to_video=r'd:\ND250fpsOct23\Cas9-B1\P1000360_stab.mp4'

# Linux Filepaths
#path_to_video=r'/media/frogtracker/Beck 07/ND250fpsOct23/Cas9-B1/P1000355.MP4'
#path_for_output = r'/home/frogtracker/Videos/split_videos'


frame_limit=10

splitter = vs(path_to_video, path_for_output)
#splitter = vs(path_to_video, path_for_output)
splitter()

