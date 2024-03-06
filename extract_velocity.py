import pandas as pd
import numpy as np
import os


filepath = r"D:\split_videos\Cas9-B1\P1000355\ts01DLC_resnet50_single_tadpoleNov23shuffle1_1030000.csv"
#path_to_data = r'd:\tracking_locations\ts04DLC_resnet50_single_tadpoleNov23shuffle1_1030000.csv'

tracked_data= pd.read_csv(filepath)

class TrainingExtractor:
    def __init__(self, directory, label_file):
        self.directory=directory
        self.label_file_path = os.path.join(directory, label_file)
        self.label_file = label_file

    def main(self):
        structured_data = self.get_structured_data(self.label_file_path)
        self.save_file_and_images(structured_data, directory=self.directory, label_file=self.label_file)
        
    def get_structured_data(self, data_file_path):
        
        if os.path.exists(data_file_path): # Check if the file exists
            labelled_data = pd.read_csv(data_file_path) 
        else:
            print(f"The file '{data_file_path}' does not exist in the specified directory.")

        # cleaning

        Na_threshold = len(labelled_data.columns) / 2 # get rid of rows where more thna half odf the values are NAs
        labelled_data = labelled_data.dropna(thresh=Na_threshold)


        labelled_data = labelled_data.T

        scorer_name = labelled_data.index[1]
        labelled_data=labelled_data.reset_index(drop=True)
        labelled_data.columns = labelled_data.iloc[0]
        labelled_data= labelled_data.drop(0)


        ######### 
        # transform to a dictionary of dictionaries of data frames. Outer dict is file names to sets of individuals
        # inner dictionary is sets of individuals to sets of coordinates for body parts
        # Set the 'individuals', 'bodyparts', and 'coords' columns as the index
        labelled_data.set_index(['individuals', 'bodyparts', 'coords'], inplace=True)
        #labelled_data.columns.levels

        file_paths = labelled_data.columns

        # Initialize an empty dictionary to store the result
        file_individual_coords_dict = {}

        # Iterate over each file path
        for file_path in file_paths:
            # Extract data for the current file path
            file_data = labelled_data[file_path]

            # Create an array to store dataframes for each individual
            individual_labelled_datas = {}

            # Iterate over individuals (ts01, ts02, ..., ts24)
            for individual in file_data.index.levels[0]:
                # Extract data for the current individual
                individual_data = file_data.loc[individual]
                #individual_data =pd.DataFrame(individual_data)
                # Pivot the dataframe to have 'bodyparts' as columns and 'coords' as index
                individual_data_pivot = individual_data.unstack(level='bodyparts')
                
                individual_data_pivot = individual_data_pivot.apply(pd.to_numeric, errors='coerce') # convert to floats
                # Add the individual dataframe to the dictionary
                individual_labelled_datas[individual] = individual_data_pivot


            # Add the array of dataframes to the dictionary with the file path as the key
            file_individual_coords_dict[file_path] = individual_labelled_datas
            
        first_file_path = next(iter(file_individual_coords_dict))

        # Get the first individual's data frames
        first_individual_dfs = file_individual_coords_dict[first_file_path]

        first_individual_first_time = next(iter(first_individual_dfs))

        # print first individual first time value
        first_individual_dfs[first_individual_first_time]
        
        return(file_individual_coords_dict)
    
    