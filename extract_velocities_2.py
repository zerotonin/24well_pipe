import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
# function to adjust eyeball posistion sif ther eis a low likelihood for the position of an eye
def adjust_eyes(left_eye_column_in, right_eye_column_in, likelihood_threshold=0.998):
    # copy data
    left_eye_column= left_eye_column_in
    right_eye_column=  right_eye_column_in
    # check same shape input data
    if left_eye_column.shape!=right_eye_column.shape:
        # print("Error, shape of right and left eye data frames differs")
        return(pd.DataFrame([]))

    for i in range(left_eye_column.shape[0]):
        # If unsure of both eyes, replace with last known position or skip if i=0
        if left_eye_column.at[i,'likelihood']<likelihood_threshold and right_eye_column.at[i,'likelihood']<likelihood_threshold:
            if i==0:
                #print("Error, unsure of both eye positions on first data point, skipping correction")
                continue 
            else:
                left_eye_column.at[i,'x']= left_eye_column[i-1,'x']
                right_eye_column.at[i,'x']= right_eye_column[i-1,'x']
                left_eye_column.at[i,'y']= left_eye_column[i-1,'y']
                right_eye_column.at[i,'y']= right_eye_column[i-1,'y']
                print("unsure of both predictions of eyes, using previous")
                continue
    
        # If unsure of left eye only ( exclusive because of guard statement above)
        if left_eye_column.at[i,'likelihood']<likelihood_threshold:
            #print("left eye unsure as likelihood is " + str(left_eye_column.at[i,'likelihood']) + "\n replacing " + str(left_eye_column.at[i,'x']) + " " + str(left_eye_column.at[i,'y']) + " with " + str(right_eye_column.at[i,'x']) + " " + str(right_eye_column.at[i,'y']))
            left_eye_column.at[i,'x']= right_eye_column.at[i,'x']
            left_eye_column.at[i,'y']= right_eye_column.at[i,'y']
            

        # If unsure of Right eye only ( exclusive because of guard statement above)
        if right_eye_column.at[i,'likelihood']<likelihood_threshold:
            #print("right eye unsure as likelihood is " + str(right_eye_column.at[i,'likelihood']) + "\n replacing " + str(right_eye_column.at[i,'x']) + " " + str(right_eye_column.at[i,'y']) + " with " + str(left_eye_column.at[i,'x']) + " " + str(left_eye_column.at[i,'y']))
            right_eye_column.at[i,'x']= left_eye_column.at[i,'x']
            right_eye_column.at[i,'y']= left_eye_column.at[i,'y']
    return (left_eye_column, right_eye_column)


def get_frons(left_eye_column_in, right_eye_column_in):
    # function to calculate frons position - frons is halfway between the 2 eyes
    frons_x= (left_eye_column_in["x"]+right_eye_column_in["x"])/2
    frons_y=(left_eye_column_in["y"]+right_eye_column_in["y"])/2
    frons_index = pd.MultiIndex.from_product([["frons"], ["x", "y"]])
    frons_df = pd.DataFrame( columns=frons_index)
    frons_df[("frons", "x")] = frons_x
    frons_df[("frons", "y")] = frons_y
    return frons_df

def get_com(left_eye_column_in, right_eye_column_in, tail_base_column_in):
    # function to calculate centre of mass of the tadpole
    com_x= (left_eye_column_in["x"]+right_eye_column_in["x"]+tail_base_column_in["x"])/3
    com_y=(left_eye_column_in["y"]+right_eye_column_in["y"]+tail_base_column_in["y"])/3
    com_index = pd.MultiIndex.from_product([["com"], ["x", "y"]])
    com_df = pd.DataFrame( columns=com_index)
    com_df[("com", "x")] = com_x
    com_df[("com", "y")] = com_y
    return com_df

def extract_xy_vectors(df):
    vecs_out=[]
    for index, row in df.iterrows():
    # Extract x and y coordinates into a tuple
        coord_tuple = (row['x'], row['y'])
        # Append the tuple to list
        vecs_out.append(coord_tuple)
    return(vecs_out)

def get_yaw(frons_col_in,tail_base_col_in):
    frons_vecs = extract_xy_vectors(frons_col_in)
    tail_base_vecs = extract_xy_vectors(tail_base_col_in)

    yaws_cartesian = []
    yaws_radians = []
    # Iterate over both lists simultaneously using zip
    for frons_vec, tail_base_vec in zip(frons_vecs, tail_base_vecs):
        # Compute the difference between corresponding vectors
        diff_x = frons_vec[0] - tail_base_vec[0]
        diff_y = frons_vec[1] - tail_base_vec[1]
        # Append the difference vector to the list
        yaws_cartesian.append((diff_x, diff_y))
        yaws_radians.append(np.arctan2(diff_y, diff_x))
    # create dataframe for yaws to be compatible with other df
    yaw_index = pd.MultiIndex.from_product([["yaw"], ["yaw_cartesian", "yaw_radians"]])
    yaw_df=pd.DataFrame( columns=yaw_index)
    yaw_df[("yaw", "yaw_cartesian")] = yaws_cartesian
    yaw_df[("yaw", "yaw_radians")] = yaws_radians
    return(yaw_df)

def get_yaw_diff(yaw_rads_in):
    yaw_rads_diff=yaw_rads_in.diff()
    yaw_rads_diff=yaw_rads_diff.drop(0)
    last_yaw=yaw_rads_diff.iloc[-1]
    last_yaw_series = pd.Series(last_yaw, index=[len(yaw_rads_diff)])
    yaw_rads_diff = pd.concat([yaw_rads_diff, last_yaw_series])
    return yaw_rads_diff.reset_index(drop=True)

def get_yaw_speed(yaw_diff_in, fps=50):
    yaw_speeds=yaw_diff_in*fps
    return yaw_speeds

def get_com_diff (com_in):
    difference_vectors = []
    # Iterate through the dataframe to calculate the difference vectors
    for i in range(1, len(com_in)):
        x_diff = com_in.loc[i, 'x'] - com_in.loc[i-1, 'x']
        y_diff = com_in.loc[i, 'y'] - com_in.loc[i-1, 'y']
        difference_vectors.append((x_diff, y_diff))
    difference_vectors.append((x_diff, y_diff)) # append last value twice
    return difference_vectors

def get_2d_rotation_matrix(yaw):
    # Define the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                                [np.sin(yaw), np.cos(yaw)]])
    return(rotation_matrix)

def get_thrust_and_slip(yaw_col_in,com_diff_col_in):
    thrusts = []
    slips = []
    rotated_com_diffs = []
    # Iterate over both lists simultaneously using zip
    for yaw, speed in zip(yaw_col_in, com_diff_col_in):
        # Compute the difference between corresponding vectors
        rot_matrix=get_2d_rotation_matrix(-yaw)
        speed= np.array(speed) # make tuple into np array
        rotated_speed = np.dot(rot_matrix, speed)
        thrusts.append(rotated_speed[0]) # add the x component of rotated speed to the thrust set
        slips.append(rotated_speed[1]) # add y component to slips list
        rotated_com_diffs.append((rotated_speed[0],rotated_speed[1]))
    return thrusts, slips, rotated_com_diffs


filepath = r"c:\Users\Lindsay\Documents\alexander\Honors\data_with_coords\Cas9-B1\P1000355\ts02DLC_resnet50_single_tadpoleNov23shuffle1_1030000.h5"

data_in = pd.read_hdf(filepath)
data=data_in.copy()
data.columns = data.columns.droplevel(level='scorer')

# adjust eyes
data['left_eye'], data['right_eye'] = adjust_eyes(data['left_eye'], data['right_eye'])

# get frons
frons_df = get_frons(data['left_eye'], data['right_eye'])
data=pd.concat([data, frons_df], axis=1)

# get yaw
yaw_df = get_yaw(data['frons'], data['tail_base'])
data=pd.concat([data, yaw_df], axis=1)
 # get yaw change
yaw_diff = get_yaw_diff(data['yaw']['yaw_radians'])
data[("yaw","yaw_diff")] = yaw_diff

# get yaw speed
yaw_speed = get_yaw_speed(data['yaw']['yaw_diff'])
data[("yaw","yaw_speed")] = yaw_speed

# get centre of mass
com_df = get_com(data['left_eye'], data['right_eye'], data['tail_base'])
data=pd.concat([data, com_df], axis=1)

# get difference in centre of mass per frame
com_diff = get_com_diff(data["com"])
data[('com',"com_diff")] = com_diff

# get thrust and slip, and the combined pair as a tuple - the rotated com diff
data[("com","thrust")], data[("com","slip")], data[("com","rotated_com_diff")] = get_thrust_and_slip(data[("yaw", "yaw_radians")],data[("com", "com_diff")])


# Plotting code

# Assuming data is your DataFrame containing columns 'x', 'y', 'yaw', 'com', 'com_diff', and 'rotated_com_diff'


idx_to_visualise = 116
next_index = idx_to_visualise + 1

# Get body part names from the top level of MultiIndex
bodyparts = data.columns.levels[0]

# Create the plot
plt.figure(figsize=(12, 6))

# Loop through each body part
for bodypart in bodyparts:
    if bodypart == 'yaw':
        continue

    # Select data for the current index and the next index
    bodypart_data_idx = data.loc[idx_to_visualise, bodypart]  # Data for idx_to_visualise
    bodypart_data_next = data.loc[next_index, bodypart]     # Data for next index

    # Extract x and y data for both indices
    x_position_idx = bodypart_data_idx['x']
    y_position_idx = bodypart_data_idx['y']
    x_position_next = bodypart_data_next['x']
    y_position_next = bodypart_data_next['y']

    # Plot the points for both indices with labels
    plt.plot(x_position_idx, y_position_idx, 'o', label=bodypart + '_idx')  # Plot idx_to_visualise
    plt.plot(x_position_next, y_position_next, 'o', label=bodypart + '_next')  # Plot next index
    plt.text(x_position_idx + 1, y_position_idx + 1, bodypart+ '_idx', fontsize=8)  # Add text label
    plt.text(x_position_next + 1, y_position_next + 1, bodypart+ '_next', fontsize=8)  # Add text label


    # Plot the yaw vector starting from tail_base for idx_to_visualise
    if bodypart == 'tail_base':  # Assuming 'tail_base' is the starting point for yaw vectors
        # Extract yaw vector data for idx_to_visualise
        yaw_vector_idx = data.loc[idx_to_visualise, ('yaw', 'yaw_cartesian')]
        # Unpack x and y components of yaw vector for idx_to_visualise
        yaw_x_idx, yaw_y_idx = yaw_vector_idx
        # Plot the yaw vector starting from tail_base point for idx_to_visualise
        plt.arrow(x_position_idx, y_position_idx, yaw_x_idx, yaw_y_idx, color='r', label='Yaw Vector_idx')

    # Plot the yaw vector starting from tail_base for next index
    if bodypart == 'tail_base':  # Assuming 'tail_base' is the starting point for yaw vectors
        # Extract yaw vector data for next index
        yaw_vector_next = data.loc[next_index, ('yaw', 'yaw_cartesian')]
        # Unpack x and y components of yaw vector for next index
        yaw_x_next, yaw_y_next = yaw_vector_next
        # Plot the yaw vector starting from tail_base point for next index
        plt.arrow(x_position_next, y_position_next, yaw_x_next, yaw_y_next, color='b', label='Yaw Vector_next')

    # Plot the center of mass difference vector
    if bodypart=='com':
        com_diff_vector = data.loc[idx_to_visualise, ('com', 'com_diff')]
        com_diff_x, com_diff_y = com_diff_vector
        plt.arrow(x_position_idx, y_position_idx, com_diff_x, com_diff_y, color='g', linestyle='--', label='COM Diff Vector')

# Configure plot
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Body Part Positions at Time Point: " + str(idx_to_visualise) + " and " + str(next_index))
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
