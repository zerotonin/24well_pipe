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
import argparse

class VideoSplitter:
    def __init__(self, path_to_video, path_for_output, frame_limit=float('inf')):
        self.path_to_video = path_to_video
        self.path_for_output = path_for_output
        self.frame_limit = frame_limit
        self.vid_capture = cv.VideoCapture(path_to_video)
        self.first_subframe_radius=0

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
        return scipy.ndimage.filters.gaussian_filter(data, sigma,axes=0,mode = 'nearest')
    
    def median_filter(self,data,fps):
        window_size = self.calculate_window(fps,nseconds=10)
        return scipy.ndimage.filters.median_filter(data, size=int(window_size),axes=0)
    
    def no_filter(self,data):
        return data
    
    def replicate_first_pointset(self,original_array):
        # Get the shape of the original array
        n, _, _ = original_array.shape

        # Create a new array by replicating the first frame along the first dimension
        replicated_array = np.tile(np.median(original_array[0:19, :, :], axis=0), (n, 1, 1))

        return replicated_array

    def filter_coordinates(self,coordinates):
        # Initialize output array with the same shape as input
        
        
        # change_in_x= np.zeros(coordinates.shape[0])
        # change_in_y= np.zeros(coordinates.shape[0])
        diff_coordinates = np.diff(coordinates,axis=0)
        mean_offset = np.mean(diff_coordinates,axis=1)
        offset_cumsum= np.cumsum(mean_offset,axis=0)
        offset_cumsum = np.tile(offset_cumsum[:, np.newaxis, :], (1, 24, 1))
        
        # for t in range(1, coordinates.shape[0]):
        #     # Calculate geometric mean change in x and y across all 24 coordinates
        #     mean_change_x = np.exp(np.mean(np.log(coordinates[t, :, 0] / coordinates[t - 1, :, 0])))
        #     mean_change_y = np.exp(np.mean(np.log(coordinates[t, :, 1] / coordinates[t - 1, :, 1])))

        #     # Update cumulative sum of geometric mean changes
        #     change_in_x[t]= mean_change_x
        #     change_in_y[t] = mean_change_y

        corrected_coordinates= self.replicate_first_pointset(coordinates)
        corrected_coordinates[1::,:,:] += offset_cumsum
        
        # cumsum_change_x=np.cumsum(change_in_x)
        # cumsum_change_y=np.cumsum(change_in_y)
        #     # Update x and y coordinates based on cumulative sums
        # filtered_array[:,:,0] += cumsum_change_x
        # filtered_array[:,:,1] += cumsum_change_y
        return corrected_coordinates
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
        n_frames_to_process=min(frame_limit, total_frames)
        #frame_splitters = list()
        with tqdm(total=n_frames_to_process, desc="Finding Wells", unit="frame") as pbar:
            while(vid_capture.isOpened()):
            # vid_capture.read() methods returns a tuple, first element is a bool 
            # and the second is frame

                if framecounter==frame_limit:
                    break
                
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
                    #print(framecounter)
                    framecounter = framecounter+1
                    pbar.update(1) 
                    ###############
                    #to display video as I go if wanted
                    # cv.imshow('Frame',frame)
                    # key = cv.waitKey(1)
                    # if key == ord('q'):
                    #     break
                    #################
                else:
                    break
        vid_capture.set(cv.CAP_PROP_POS_FRAMES, 0) # restart the video object

        framecounter=0
        subframes_edge_length= first_subframe_radius*2 # find size of the subframes - radius times 2, plus 1 for the 0 column
        raw_subframe_centres_arr = np.array(raw_subframe_centres)
        #smoothed_subframe_centres = self.butter_lowpass_filter(raw_subframe_centres_arr, cutoff=0.3, fs=fps, order=4)
        #smoothed_subframe_centres = self.no_filter(raw_subframe_centres_arr.squeeze())
        
        # with open('raw_centres.npy', 'wb') as f:
        #     np.save(f, smoothed_subframe_centres)
        
        #smoothed_subframe_centres =self.gaussian_filter(raw_subframe_centres_arr.squeeze(),fps)
        #smoothed_subframe_centres =self.median_filter(raw_subframe_centres_arr.squeeze(),fps)
        smoothed_subframe_centres =self.filter_coordinates(raw_subframe_centres_arr.squeeze())
        smoothed_subframe_centres =self.gaussian_filter(smoothed_subframe_centres,fps)
        #smoothed_subframe_centres = self.butter_lowpass_filter(smoothed_subframe_centres, cutoff=1, fs=fps, order=4)
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
        self.first_subframe_radius=first_subframe_radius
        #print("Resources released.")
        
    def change_file(self, new_video_filepath):
        self.path_to_video = new_video_filepath
        self.vid_capture = cv.VideoCapture(new_video_filepath)

    def get_well_radius(self):
        return(self.first_subframe_radius)

    def __call__(self):
        self.main()
        
    def main(self):
        self.create_and_write_subframes(path_to_video=self.path_to_video, path_for_output=self.path_for_output,
                            vid_capture=self.vid_capture,frame_limit=self.frame_limit)
# Create a video capture object, in this case we are reading the video from a file

# next find a way to call the get subframe radius