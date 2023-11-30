import cv2 as cv
from frame_splitter import frameSplitter
import scipy
from scipy.signal import filtfilt, butter
import numpy as np
import os
import re
import shutil
from tqdm import tqdm
from pystackreg import StackReg
import pickle as pkl


class VideoSplitter:
    def __init__(self, path_to_video, path_for_output, frame_limit=float('inf')):
        self.path_to_video = path_to_video
        self.path_for_output = path_for_output
        self.frame_limit = frame_limit
        self.vid_capture = cv.VideoCapture(path_to_video)

    def calculate_window(self, fps, nseconds=2):
        window = fps*nseconds
        return window

    def butter_lowpass_filter(self,data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data, axis=0)
        return y

    def gaussian_filter(self,data,fps):
        sigma = self.calculate_window(fps,nseconds=2)
        return scipy.ndimage.filters.gaussian_filter(data, sigma,axes=0)
    
    def median_filter(self,data,fps):
        window_size = self.calculate_window(fps,nseconds=10)
        return scipy.ndimage.filters.median_filter(data, size=int(window_size),axes=0)
    
    def no_filter(self,data):
        return data
    
    def filter_coordinates(self,coordinates, threshold=3):
        n, _, _ = coordinates.shape

        # Initialize the filtered coordinates array
        filtered_coordinates = np.zeros_like(coordinates)

        # Iterate over time points
        for t in range(n):
            if  t==0:
                filtered_coordinates[t, :, 0] = coordinates[t , :, 0]
                filtered_coordinates[t, :, 1] = coordinates[t , :, 1]
                continue
            # Calculate the median change in x and y
            median_change_x = np.median(coordinates[t, :, 0] - coordinates[t - 1, :, 0])
            median_change_y = np.median(coordinates[t, :, 1] - coordinates[t - 1, :, 1])
            print(f"Time {t}: Median Change (x, y) = ({median_change_x}, {median_change_y})")
            # Check if the median change exceeds the threshold
            if abs(median_change_x) > threshold:
                # Update x coordinates based on the median change
                filtered_coordinates[t, :, 0] = filtered_coordinates[t - 1, :, 0] + median_change_x
            else:
                # Keep x coordinates the same as the previous time point
                filtered_coordinates[t, :, 0] = filtered_coordinates[t - 1, :, 0]

            if abs(median_change_y) > threshold:
                # Update y coordinates based on the median change
                filtered_coordinates[t, :, 1] = filtered_coordinates[t - 1, :, 1]+ median_change_y
            else:
                # Keep y coordinates the same as the previous time point
                filtered_coordinates[t, :, 1] = filtered_coordinates[t - 1, :, 1]

        return filtered_coordinates


    def create_folder_for_video(self, path_to_video, output_directory):
        # Extracting the numbers before ".MP4" in the file name

        filename = os.path.basename(path_to_video).split('.')[0]
        foldername = os.path.basename(os.path.dirname(path_to_video))

        super_folder_path= os.path.join(os.path.join(output_directory,foldername),filename)

        if not os.path.exists(super_folder_path): # add folder for split images
            os.makedirs(super_folder_path, exist_ok =True)
        else:
            shutil.rmtree(super_folder_path) # if already exists, clear the directory
            os.mkdir(super_folder_path)
        
        return(super_folder_path)

    def initialize_video_writers(self, width, height, fps, initial_video_directory, output_directory):

        output_folder = self.create_folder_for_video(initial_video_directory, output_directory)
        writers = []
        for i in range(1, 25):
            # Define the codec and create a VideoWriter object
            individual_output_path = os.path.join(output_folder, f"ts{i:02}.mp4")
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            writer = cv.VideoWriter(individual_output_path, fourcc, int(fps), (int(width), int(height)))
            writers.append(writer)
        return writers

    # Function to release VideoWriters
    def release_video_writers(self, writers):
        for writer in writers:
            writer.release()
            
    def crop_image(self, img,y_center,x_center,radius = 100):

        y_min =  int(int(y_center)-radius)
        y_max =  int(int(y_center)+radius)

        x_min=int(int(x_center)-radius)
        x_max =  int(int(x_center)+radius)
        
        radius = int(radius)
        
        if  y_min<0:
            y_min = 0
        if y_max > img.shape[1]:
            y_max = img.shape[1]
            
        if  x_min<0:
            x_min = 0
        if x_max > img.shape[0]:
            x_max = img.shape[0]
        
        return img[x_min:x_max,y_min:y_max]
        
    def crop_filtered_centres(self, img, filtered_centres, median_radius):
        cropped_images= [] 
        if filtered_centres is None:
            print("Error, filtered centres not provided")
            return None
        
        for circle in filtered_centres:
            cropped_img=self.crop_image(img,x_center=circle[1], y_center=circle[0], radius=median_radius)
            cropped_images.append(cropped_img)
        return cropped_images
            

    def create_and_write_subframes(self, path_to_video, path_for_output,vid_capture,frame_limit):
        if (vid_capture.isOpened() == False):
            print("Error opening the video file", path_to_video)
            return()
            # Read fps and frame count
        else:
            # Get frame rate information
            # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
            fps = vid_capture.get(5)
            #print('Frames per second : ', fps,'FPS')

            # Get frame count
            # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
            total_frames = vid_capture.get(7)
            #print('Frame count : ', total_frames)

        raw_subframe_centres = list()
        framecounter=0
        n_frames_to_process=10
        frames_to_sample=np.linspace(0,total_frames, n_frames_to_process)
        
        frame_nos = frames_to_sample/total_frames
        for frame_no in frame_nos:
            vid_capture.set(2,frame_no)
            ret, frame = vid_capture.read()
            if ret == True:
                if framecounter==0:
                    fs = frameSplitter(frame)
                    first_subframe_radius = fs.process(mode='radius') # Find radius of circles in firt subframe o that this can remain constant throughout video otherwisse writers wont work
                else: fs=frameSplitter(frame, first_frame_radius=first_subframe_radius)
                #frame_splitters.append(fs)
                if fs.process(mode='centres').all()!=None: # check that it can find centres
                    centres = fs.process(mode='centres')
                elif len(raw_subframe_centres) > 0:
                    print("Couldnt find centres, so using previously known centres for the following frame:")
                    centres = raw_subframe_centres[-1].copy()  # Use copy previously known centres
                else: print("Error, Couldn't find wells for the first frame of the video. ")
                raw_subframe_centres.append(centres)
            framecounter+=1
        vid_capture.set(cv.CAP_PROP_POS_FRAMES, 0) # restart the video object
        geometric_mean_location= np.exp(np.mean(np.log(raw_subframe_centres.squeeze()), axis=0))

        smoothed_subframe_centres = np.tile(geometric_mean_location, (total_frames, 1, 1))
        
        framecounter=0
        subframes_edge_length= first_subframe_radius*2 # find size of the subframes - radius times 2, plus 1 for the 0 column
        # raw_subframe_centres_arr = np.array(raw_subframe_centres)
        #smoothed_subframe_centres = self.butter_lowpass_filter(raw_subframe_centres_arr, cutoff=0.3, fs=fps, order=4)
        #smoothed_subframe_centres = self.no_filter(raw_subframe_centres_arr.squeeze())
        
        # with open('raw_centres.npy', 'wb') as f:
        #     np.save(f, smoothed_subframe_centres)
        
        #smoothed_subframe_centres =self.gaussian_filter(raw_subframe_centres_arr.squeeze(),fps)
        #smoothed_subframe_centres =self.median_filter(raw_subframe_centres_arr.squeeze(),fps)
        #smoothed_subframe_centres =self.filter_coordinates(raw_subframe_centres_arr.squeeze())
        #smoothed_subframe_centres =self.stabilize_frames(raw_subframe_centres_arr.squeeze(),fps)
        smoothed_subframe_centres = [ smoothed_subframe_centres[i,:,:].squeeze() for i in range(smoothed_subframe_centres.shape[0])]

        video_writers = self.initialize_video_writers(width=subframes_edge_length, 
                                                height=subframes_edge_length, 
                                                fps=fps, 
                                                initial_video_directory = path_to_video, 
                                                output_directory = path_for_output)

        # print("Initialised Height and width as ",int(subframes_edge_length) )
        #video_writers = initialize_video_writers(width=vid_capture.get(cv.CAP_PROP_FRAME_WIDTH), height=vid_capture.get(cv.CAP_PROP_FRAME_HEIGHT), fps=fps, initial_video_directory = path_to_video, output_directory = path_for_output)
        #print(f"Number of video writers: {len(video_writers)}")
        #print("writing videos")
        with tqdm(total=n_frames_to_process, desc="Writing Individual Wells", unit="frame") as pbar:
            while(vid_capture.isOpened()):
                ret, frame = vid_capture.read()
                if ret and framecounter<frame_limit:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    subframes = self.crop_filtered_centres(frame,smoothed_subframe_centres[framecounter],first_subframe_radius)
                    for i, subframe in enumerate(subframes):
                    #to display video as I go if wanted
                        #height, width = subframe.shape[:2]
                        # print("h:",height,"w",width,"\n")
                        #if height != width or width != subframes_edge_length or height != subframes_edge_length:
                            # print(height, width, subframes_edge_length, "not equal for frame:", framecounter, "video", i)
                        # cv.imshow('Frame',subframe)
                        # key = cv.waitKey(0)
                        # if key == ord('q'):
                        #     break
                        video_writers[i].write(cv.cvtColor(subframe, cv.COLOR_GRAY2BGR))
                        #  print(f"Frame {framecounter} written to video {i}")
                    framecounter = framecounter+1
                    pbar.update(1) 

                else:
                    break
            
        

        vid_capture.release()
        self.release_video_writers(video_writers)
        cv.destroyAllWindows()
        #print("Resources released.")
        
    def change_file(self, new_video_filepath):
        self.path_to_video = new_video_filepath
        self.vid_capture = cv.VideoCapture(new_video_filepath)

    def __call__(self):
        self.main()
    
    def main(self):
        self.create_and_write_subframes(path_to_video=self.path_to_video, path_for_output=self.path_for_output,
                            vid_capture=self.vid_capture,frame_limit=self.frame_limit)
# Create a video capture object, in this case we are reading the video from a file





