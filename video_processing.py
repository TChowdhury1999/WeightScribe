import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import string
import re
import datetime as dt
import pandas as pd


"""
Contains main functions for use in the app

extract_weight_df() takes the video path and saves the output df to output_path
"""

def get_motion_arr(video_path):
    """
    Return an np array of "motion" from the video at the provided path
    """

        # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize variables for accumulating differences
    accumulated_diff = None
    frame_count = 0

    # Read the first frame
    ret, prev_frame = cap.read()

    # Check if the frame is read correctly
    if not ret:
        print("Error reading the first frame")

    while ret:
        # Read the next frame
        ret, current_frame = cap.read()
        
        if not ret:
            break
        
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute the absolute difference between the current frame and the previous frame
        diff_frame = cv2.absdiff(current_gray, prev_gray)
        
        # Accumulate the difference
        if accumulated_diff is None:
            accumulated_diff = np.zeros_like(diff_frame, dtype=np.float32)
        
        accumulated_diff += diff_frame
        frame_count += 1
        
        # Set the current frame as the previous frame for the next iteration
        prev_frame = current_frame

    # Release the video capture object
    cap.release()

    # Convert the average difference to uint8
    motion_2d_arr = cv2.normalize(accumulated_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return(motion_2d_arr)

def get_run_indices(motion_2d_arr):
    """
    Returns the y indices between which the date and weights could be 
    """

    # collapse to 1d array around the centre 50%
    x_start = motion_2d_arr.shape[1] // 4 
    x_end =  motion_2d_arr.shape[1] * 3 // 4
    motion_2d_arr_middle = motion_2d_arr[:, x_start : x_end]
    motion_1d_arr = np.mean(motion_2d_arr_middle, axis=1).astype(float)

    # the most motion happens at the middle line
    # we know the date is definitely above so lets filter to before then
    # also we know the width of the filter line is about 100 pix maximum for 2400 pix video
    motion_1d_arr_top = motion_1d_arr[:np.argmax(motion_1d_arr)- int(2400*100/len(motion_1d_arr))]

    # binarize
    motion_1d_arr_binary = (motion_1d_arr_top > 0).astype(int)

    # get the start and end indices of each run
    diff = np.diff(motion_1d_arr_binary)
    start_indices = np.where(diff==1)[0] + 1
    end_indices = np.where(diff==-1)[0]
    run_indices = list(zip(start_indices, end_indices))

    return run_indices

def get_percentage_frame(video_path, percentage=0.5):
    """
    Get the % frame of video at path
    """

    # now lets open up the % frame of the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames*percentage))
    ret, frame = cap.read()
    cap.release()

    return(frame)

def get_min_x(ocr_entry):
    bounding_box = ocr_entry[0]
    x_coords = [point[0] for point in bounding_box]
    return min(x_coords)

def get_runs(reader, frame, motion_2d_arr, run_indices):
    """
    Identify which indices have weight / date info
    """

    # identify where the date and where the weight is written
    date_found = False
    date_run = []
    weight_found = False
    weight_run = []

    remove_punctuation = str.maketrans('', '', string.punctuation)
    date_pattern = r'^\d{1,2} [A-Za-z]{3}( \d{4})? \d{4}$'

    x_start = motion_2d_arr.shape[1] // 4 
    x_end =  motion_2d_arr.shape[1] * 3 // 4

    for run in run_indices:
        if run[0]<3:
            cropped_frame = frame[run[0]:run[1]+6, x_start:x_end]
        elif run[1]+6>frame.shape[0]:
            cropped_frame = frame[run[0]-3:frame.shape[0], x_start:x_end]
        cropped_frame = frame[run[0]-3:run[1]+6, x_start:x_end]
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        text = reader.readtext(gray_frame)

        # if no text is found we move on
        if len(text) == 0:
            continue

        # text is a list of tuples. tuples have 3 parts

        # for the date it is DD Mmm (YYYY), HH:MM
        if not date_found:
            sorted_ocr_output = sorted(text, key=get_min_x)
            sorted_texts = [entry[1].translate(remove_punctuation) for entry in sorted_ocr_output]
            no_punc_text = " ".join(sorted_texts)
            match = re.match(date_pattern, no_punc_text)
            if bool(match):
                # date run has been found
                date_found = True
                date_run = run
        
        # for the weight, the second tuple should have kg as the middle element
        if not weight_found:
            sorted_ocr_output = sorted(text, key=get_min_x)
            sorted_texts = [entry[1].translate(remove_punctuation) for entry in sorted_ocr_output]
            no_punc_text = " ".join(sorted_texts)
            match = re.search(r"kg", no_punc_text)
            if bool(match):
                # weight run has been found
                weight_found = True
                weight_run = run

    return(date_run, weight_run)

def get_runs_to_completion(video_path, reader, motion_2d_arr, run_indices):
    """
    Runs get_runs until date and weight runs are found
    """

    weight_run = []
    date_run = []

    current_percentage = 1

    iterations = 0
    max_iterations = 10

    while len(weight_run) == 0 or len(date_run) == 0:

        current_percentage = current_percentage / 2

        frame = get_percentage_frame(video_path, percentage=current_percentage)

        new_date_run, new_weight_run = get_runs(reader, frame, motion_2d_arr, run_indices)

        if len(new_date_run) > 0:
            date_run = new_date_run

        if len(new_weight_run) > 0:
            weight_run = new_weight_run
        
        iterations += 1
        if iterations > max_iterations:
            break

    return(date_run, weight_run)

def get_info(video_path, reader, motion_2d_arr, date_run, weight_run):
    """
    Loop through video frames and get the weight and date info
    """

    # store date and weight info below
    dates = []
    weights = []

    last_date = None
    first_frame_done = False
    prev_frame = []

    differences = []
    frames_checked = 0

    x_start = motion_2d_arr.shape[1] // 4 
    x_end =  motion_2d_arr.shape[1] * 3 // 4

    remove_punctuation = str.maketrans('', '', string.punctuation)
    date_pattern = r'^\d{1,2} [A-Za-z]{3}( \d{4})? \d{4}$'

    # loop through each frame

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # crop frame to date portion
        if date_run[0]<3:
            cropped_frame = frame[date_run[0]:date_run[1]+6, x_start:x_end]
        elif date_run[1]+6>frame.shape[0]:
            cropped_frame = frame[date_run[0]-3:frame.shape[0], x_start:x_end]
        cropped_frame = frame[date_run[0]-3:date_run[1]+6, x_start:x_end]
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)


        if first_frame_done == False:
            # if its the first frame we move on
            prev_frame = gray_frame
            first_frame_done = True
            continue
        
        # lets see if the frame has changed
        difference = (gray_frame - prev_frame).sum()
        differences.append(difference)
        prev_frame = gray_frame
        
        # if the difference is less than 4,000 then we can assume the frame has remained the same
        if difference < 4_000:
            continue
        
        frames_checked +=1

        # expand the frame
        new_height, new_width = 3 * np.array(gray_frame.shape[:2])
        expanded_frame = cv2.resize(gray_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # remove white space
        # Threshold the grayscale image to get a binary image
        _, binary = cv2.threshold(expanded_frame, 240, 255, cv2.THRESH_BINARY_INV)

        # Find contours and get the bounding box of the largest contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))

        # Crop the image using the bounding box
        final_frame_to_read = expanded_frame[y:y+h, x:x+w]

        text = reader.readtext(final_frame_to_read)
        
        # check if text detected
        if len(text)>0:

            # remove any punctuation
            sorted_ocr_output = sorted(text, key=get_min_x)
            sorted_texts = [entry[1].translate(remove_punctuation) for entry in sorted_ocr_output]
            no_punc_text = " ".join(sorted_texts)

            # does it match expected format?
            match = re.match(date_pattern, no_punc_text)
            if not bool(match):
                continue

            # do we already have this date?
            if no_punc_text not in dates:
                dates.append(no_punc_text)
            else:
                continue

        # record the weight 
        weight_cropped = frame[weight_run[0]-3:weight_run[1]+6, x_start:x_end]
        gray_weight = cv2.cvtColor(weight_cropped, cv2.COLOR_BGR2GRAY)
        weight_text = reader.readtext(gray_weight)

        if len(weight_text)>0:
            weights.append(weight_text[0][1])
        else:
            weights.append(np.nan)

    cap.release()

    weights = [float(i[:-3]) for i in weights]

    return(dates, weights)

def package_to_df(dates, weights):
    """
    Process date info into timestamps and then packages into a pandas dataframe
    """
    timestamps = []
    for date in dates:
        date_parts = date.split(" ")

        # add in year
        if len(date_parts) == 3:
            date_parts.insert(-1, str(dt.datetime.now().year))
        
        # convert back to str
        date_str = f"{date_parts[0]} {date_parts[1]} {date_parts[2]} {date_parts[3][:2]}:{date_parts[3][2:]}"

        # convert to timestamp
        timestamp = dt.datetime.strptime(date_str, "%d %b %Y %H:%M")

        timestamps.append(timestamp)

    return (pd.DataFrame({"datetime":timestamps, "weight": weights}))

def prettify_df(df):
    df["date"] = pd.to_datetime(df["datetime"]).dt.strftime('%a %d %b %Y')

    # remove current year
    current_year = dt.datetime.now().year
    df["date"] = df["date"].apply(lambda x: x[:-5] if x[-4:] == str(current_year) else x)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df['rolling_avg'] = df['weight'].rolling(window='7D').mean().round(2)

    return df

def extract_weight_df(video_path, output_path, filename):

    motion_2d_arr = get_motion_arr(video_path)
    run_indices = get_run_indices(motion_2d_arr)
    reader = easyocr.Reader(['en'])
    date_run, weight_run = get_runs_to_completion(video_path, reader, motion_2d_arr, run_indices)
    dates, weights = get_info(video_path, reader, motion_2d_arr, date_run, weight_run)
    weight_df = package_to_df(dates, weights)
    weight_df = prettify_df(weight_df)
    weight_df.to_csv(output_path+"/"+filename+".csv", index=False)

    return()