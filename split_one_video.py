from video_splitter import VideoSplitter as vs
import os

# for stab
# path_for_output = r'd:\ND250fpsOct23\stabilised_first_then_split'
# path_to_video=r'd:\ND250fpsOct23\Cas9-B1\P1000360_stab.mp4'

# Windows Filepaths for non stab
path_for_output = r'd:\ND250fpsOct23\split_videos_custom_filter'
path_to_video=r'd:\ND250fpsOct23\ND2-A1\P1030820.MP4'

# Linux Filepaths
#path_to_video=r'/media/frogtracker/Beck 07/ND250fpsOct23/Cas9-B1/P1000355.MP4'
#path_for_output = r'/home/frogtracker/Videos/split_videos'


frame_limit=1500

splitter = vs(path_to_video, path_for_output,frame_limit=frame_limit)
#splitter = vs(path_to_video, path_for_output)
splitter()

