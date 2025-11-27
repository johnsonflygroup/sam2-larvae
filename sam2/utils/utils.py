import copy
import math
import os
from collections import defaultdict
from itertools import combinations

# Set environment variable to disable ultralytics verbose logging
os.environ['YOLO_VERBOSE'] = "False"

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter
from scipy.spatial import distance
from shapely.geometry import Polygon
from tqdm import tqdm
from ultralytics.engine.results import Masks
from ultralytics.utils.plotting import Annotator, colors


def get_video_info(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Retrieve video properties: width, height, and frames per second
    w, h, fps, frame_count = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return w, h, fps, frame_count


def get_larva_detections(video_path, detection_model, steps=30, num_larvae=5):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Check if video file exists
    if not cap.isOpened():
        print("Error opening video file")

    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'Looking for {num_larvae} larvae in video')

    # Set frame position
    for frame_index in range(0, total_frames, steps):
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        results = detection_model.predict(frame, verbose=False)  # return a list of Results objects
        result = results[0]

        boxes = result.boxes
        boxes_xyxy = boxes.xyxy.cpu().numpy()

        num_detections = len(boxes_xyxy)
        
        if num_detections == num_larvae:
            print(f'Found {num_larvae} larvae at frame {frame_index}.')
            return result, boxes_xyxy, frame_index
        else:
            continue

    # Release resources
    cap.release()

    print(f'Failed to predict {num_larvae} larvae in a frame, found {num_detections} larvae at frame {frame_index}')

    return result, boxes_xyxy, frame_index


def extract_frames_ffmpeg(video_path, output_dir, quality=2, start_number=0, fps=None, file_pattern='%05d.jpg'):
    """
    Uses ffmpeg-python to extract frames from a video and save as images.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save the extracted frames.
        quality (int): Quality level for the output images (lower is better quality).
        start_number (int): Starting number for the output image filenames.
        fps (int/None): The framerate to extract frames at.
        file_pattern (str): Pattern for naming the output images (default: '%05d.jpg').
    
    Returns:
        None
    """
    if os.path.exists(output_dir):
        print(f'{output_dir} exists')
        return
    os.makedirs(output_dir, exist_ok=True)

    # Construct output file path pattern
    output_path = f'{output_dir}/{file_pattern}'

    # Use ffmpeg to extract frames
    stream = ffmpeg.input(video_path)
    if fps is not None:
        stream = stream.filter('fps', fps=fps, round='up')
    stream.output(
        output_path, 
        q=quality,  # Quality for the frames
        start_number=start_number  # Start number for frame file names
    ).run(quiet=True)


def fit_ellipse_to_mask(mask):
    """
    Fits an ellipse to a boolean mask where the object is located.

    Parameters:
    - mask: numpy array, a boolean mask with True values representing the object.

    Returns:
    - center: tuple (x, y) representing the center of the ellipse.
    - axes: tuple (major_axis_length, minor_axis_length) representing the lengths of the major and minor axes.
    - angle: float, angle of rotation of the ellipse in degrees.
    """
    # Ensure mask is in the correct format for OpenCV
    mask = mask.squeeze().astype(np.uint8) * 255  # Convert boolean mask to 0 and 255 format

    # Find contours of the object in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contour was found
    if len(contours) == 0:
        raise ValueError("No object found in the mask to fit an ellipse.")

    # Find the largest contour by area (assume it's the object)
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit an ellipse to the largest contour
    if len(largest_contour) < 5:
        raise ValueError("Not enough points to fit an ellipse.")

    ellipse = cv2.fitEllipse(largest_contour)

    # Extract parameters for drawing
    center = (int(ellipse[0][0]), int(ellipse[0][1]))  # Center (x, y)
    axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))  # Half lengths of major and minor axes
    angle = ellipse[2]  # Rotation angle

    # Swap axes if major axis is shorter than minor axis
    if axes[0] < axes[1]:
        axes = (axes[1], axes[0])  # Swap major and minor axes
        angle += 90  # Adjust angle accordingly
        angle %= 180  # Ensure angle stays within [0, 180) range
    
    angle_x_axis = 180 - angle

    return center, axes, angle, angle_x_axis


def get_track_data(mask, h, w):
    track_data = {}

    try:
        # for drawing predictions on video
        line_mask = Masks(torch.tensor(mask), (h, w)).xy[0]

        track_data['line mask'] = line_mask
        # track_data['bbox_xyxy'] = bbox_xyxy
        centroid_point = Polygon(line_mask).centroid

        # for collecting data
        size_in_pixels = mask.sum()

        if not centroid_point.is_empty:
            track_data['centroid x'] = int(centroid_point.x)
            track_data['centroid y'] = int(centroid_point.y)
        else:
            track_data['centroid x'] = None
            track_data['centroid y'] = None
        track_data['polygon size (pixels)'] = size_in_pixels

        center, axes, angle, angle_x_axis = fit_ellipse_to_mask(mask)
        track_data['ellipse major/minor (ratio)'] = axes[0] / axes[1]
        track_data['ellipse major axis (pixels)'] = axes[0] * 2
        track_data['ellipse minor axis (pixels)'] = axes[1] * 2
        track_data['ellipse angle (degrees)'] = angle_x_axis
    except:
        track_data['line mask'] = None
        track_data['centroid x'] = None
        track_data['centroid y'] = None
        track_data['polygon size (pixels)'] = None
        track_data['ellipse major/minor (ratio)'] = None
        track_data['ellipse major axis (pixels)'] = None
        track_data['ellipse minor axis (pixels)'] = None
        track_data['ellipse angle (degrees)'] = None
        return track_data

    return track_data


def get_video_segments(predictor, inference_state, h, w):
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results

    # predict forwards
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=False):
        data = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            track_data = get_track_data(mask, h, w)

            # add data to an object
            data[out_obj_id] = track_data
        
        # add data to a frame
        video_segments[out_frame_idx] = data

    # predict backwards
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        data = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            track_data = get_track_data(mask, h, w)

            # add data to an object
            data[out_obj_id] = track_data
        
        # add data to a frame
        video_segments[out_frame_idx] = data

    video_segments = dict(sorted(video_segments.items()))  # sort by frame number

    def process_video_segments(video_segments):
        """
        Process video segments dictionary by replacing missing or invalid values 
        with the information from the previous frame of the same object.

        Args:
        video_segments (dict): Dictionary containing video segments information.

        Returns:
        dict: Processed video segments dictionary.
        """

        # Initialize an empty dictionary to store the previous frame information for each object
        previous_frame_info = {}

        # Iterate over each frame in the video segments
        for frame_number, frame_info in video_segments.items():
            # Iterate over each object in the frame
            for object_id, object_info in frame_info.items():
                # Check if object info is empty
                if object_info:
                    # If 'centroid x' or 'centroid y' is None or 'size (pixels)' is 0, 
                    # replace with the information from previous frame of the same object
                    if object_info['centroid x'] is None or object_info['centroid y'] is None or object_info['polygon size (pixels)'] == 0 \
                        or object_info['ellipse angle (degrees)'] is None:
                        video_segments[frame_number][object_id] = previous_frame_info[object_id]

                    # Update previous frame information for the object
                    previous_frame_info[object_id] = video_segments[frame_number][object_id].copy()

        return video_segments
    
    processed_video_segments = process_video_segments(video_segments)

    return processed_video_segments


def get_frame_data_subset(frame_dict, step=1):
    """
    Convert frame numbers to data per step (similar to frame number).
    
    Args:
        frame_dict (dict): Dictionary with frame numbers as keys.
        step (int): Step size.
    
    Returns:
        dict: Dictionary with steps as keys.
    """
    data = {}
    for frame, value in frame_dict.items():
        second = frame // step
        if second not in data:
            data[second] = value
    return data


def extract_obj_ids(data):
    """
    Extract all object IDs from the given data across all frames.
    
    Args:
    - data (dict): Data with structure:
                   {frame_number: {obj_id: {column_name: value, ...}, ...}, ...}
    
    Returns:
    - obj_ids (set): A set of unique object IDs found in the data.
    """
    obj_ids = set()
    
    # Traverse through the data to collect object IDs
    for frame, objects in data.items():
        obj_ids.update(objects.keys())
    
    return obj_ids


def add_raw_data(data_obj, fps=30, scale_factor=0.05, std_threshold=0.5):
    data = copy.deepcopy(data_obj)
    
    # initialize data for frame 0
    first_frame = list(data.keys())[0]
    obj_ids = extract_obj_ids(data)

    aggregated_data = defaultdict(lambda: {})

    for obj_id in obj_ids:
        major_minor_ratio = np.array([item[obj_id]['ellipse major/minor (ratio)'] for index, item in data.items()])
        mean = np.mean(major_minor_ratio)
        std = np.std(major_minor_ratio)
        aggregated_data[obj_id]['ellipse major/minor (ratio) mean'] = mean
        aggregated_data[obj_id]['ellipse major/minor (ratio) std'] = std

    # propagate across frames
    for frame_index in range(0, len(data)):
        for obj_id in obj_ids:
            # calculated data from previous frame
            previous_frame_index = frame_index - 1

            if previous_frame_index < 0:        # First frame
                # get the frame
                data[frame_index][obj_id]['frame'] = 0

                # calculate second
                data[frame_index][obj_id]['second'] = data[frame_index][obj_id]['frame'] / fps

                data[frame_index][obj_id]['polygon size (mm2)'] = data[frame_index][obj_id]['polygon size (pixels)'] * (scale_factor**2)

                # calculate major/minor ratio z-score (based on defined threshold)
                z_score = (aggregated_data[obj_id]['ellipse major/minor (ratio) mean'] - data[frame_index][obj_id]['ellipse major/minor (ratio)']) / aggregated_data[obj_id]['ellipse major/minor (ratio) std']
                if data[frame_index][obj_id]['ellipse major/minor (ratio)'] < aggregated_data[obj_id]['ellipse major/minor (ratio) mean']:
                    data[frame_index][obj_id]['major/minor ratio z-score (based on defined threshold)'] = z_score
                else:
                    data[frame_index][obj_id]['major/minor ratio z-score (based on defined threshold)'] = 0

                # check if is elongated
                if data[frame_index][obj_id]['major/minor ratio z-score (based on defined threshold)'] > std_threshold:
                    data[frame_index][obj_id]['is elongated'] = 0
                else:
                    data[frame_index][obj_id]['is elongated'] = 1

                data[frame_index][obj_id]['distance (pixels)'] = 0
                data[frame_index][obj_id]['distance (mm)'] = 0
                data[frame_index][obj_id]['speed (pixels/frame)'] = 0
                data[frame_index][obj_id]['speed (mm/second)'] = 0

                continue

            current_frame_data = data[frame_index][obj_id]
            previous_frame_data = data[previous_frame_index][obj_id]

            current_point = current_frame_data['centroid x'], current_frame_data['centroid y']
            previous_point = previous_frame_data['centroid x'], previous_frame_data['centroid y']

            # get the frame
            current_frame_data['frame'] = frame_index

            # calculate second
            second = frame_index / fps
            current_frame_data['second'] = second

            # calculate size
            current_frame_data['polygon size (mm2)'] = current_frame_data['polygon size (pixels)'] * (scale_factor**2)

            # calculate major/minor ratio z-score (based on defined threshold)
            z_score = (aggregated_data[obj_id]['ellipse major/minor (ratio) mean'] - current_frame_data['ellipse major/minor (ratio)']) / aggregated_data[obj_id]['ellipse major/minor (ratio) std']
            if current_frame_data['ellipse major/minor (ratio)'] < aggregated_data[obj_id]['ellipse major/minor (ratio) mean']:
                current_frame_data['major/minor ratio z-score (based on defined threshold)'] = z_score
            else:
                current_frame_data['major/minor ratio z-score (based on defined threshold)'] = 0

            # check if is elongated
            if current_frame_data['major/minor ratio z-score (based on defined threshold)'] > std_threshold:
                current_frame_data['is elongated'] = 0
            else:
                current_frame_data['is elongated'] = 1          

            # calculate distance
            dist = distance.euclidean(current_point, previous_point)
            current_frame_data['distance (pixels)'] = dist
            current_frame_data['distance (mm)'] = current_frame_data['distance (pixels)'] * scale_factor

            # calculate speed
            speed = current_frame_data['distance (pixels)']
            current_frame_data['speed (pixels/frame)'] = speed
            current_frame_data['speed (mm/second)'] = speed * fps * scale_factor

    return data


def process_raw_data_old(data, speed_outlier_std=5, size_outlier_std=4):
    """
    Process raw data to detect outliers in 'speed (mm/second)' and 'polygon size (mm2)' for each object across frames.

    TODO: Make outlier args actually optional

    Args:
    - data (dict): Nested dictionary with structure:
                   {frame_number: {obj_id: {'speed (mm/second)': value, ...}, ...}, ...}
    - speed_outlier_std (int, optional): Number of standard deviations to consider a value an outlier (either side of mean). Defaults to 4.
    - size_outlier_std (int, optional): Number of standard deviations to consider a value an outlier (either side of mean). Defaults to 4.
    
    Returns:
    - good_data (dict): Data without the outliers.
    - bad_data (dict): Data containing only the outliers.
    """
    # Initialize containers for good and bad data
    good_data = {}
    bad_data = {}
    
    # Extract speeds and sizes for each obj_id across frames
    obj_measures = {}
    for frame, objects in data.items():
        for obj_id, obj_data in objects.items():
            speed = obj_data.get('speed (mm/second)', None)
            size = obj_data.get('polygon size (mm2)', None)
            if speed is not None and size is not None:
                obj_measures.setdefault(obj_id, []).append((frame, speed, size))
    
    # Detect outliers for each object
    for obj_id, obj_measure in obj_measures.items():
        frames, speeds, sizes = zip(*obj_measure)  # Separate frames, speeds, and sizes
        speeds = np.array(speeds)  # Convert speeds to NumPy array
        sizes = np.array(sizes)
        
        # Detect outliers using the provided function
        speed_outliers, speed_outlier_indices = detect_outliers(speeds, speed_outlier_std)
        size_outliers, size_outlier_indices = detect_outliers(sizes, size_outlier_std)

        # Combine outliers together (union together)
        outlier_indices = np.union1d(speed_outlier_indices, size_outlier_indices)

        # Separate good and bad data
        good_frames = [frames[i] for i in range(len(speeds)) if i not in outlier_indices]
        bad_frames = [frames[i] for i in outlier_indices]
        
        # Populate good_data
        for frame in good_frames:
            good_data.setdefault(frame, {}).setdefault(obj_id, {}).update(data[frame][obj_id])
        
        # Populate bad_data
        for frame in bad_frames:
            bad_data.setdefault(frame, {}).setdefault(obj_id, {}).update(data[frame][obj_id])
    
    return good_data, bad_data


def detect_outliers(arr, threshold=4):
    """
    Detect outliers in a NumPy array using mean and standard deviation.

    Args:
    - arr (numpy.ndarray): Input array.
    - threshold (int): Number of standard deviations from mean to consider a value an outlier. Defaults to 4.

    Returns:
    - outliers (numpy.ndarray): Array of outlier values.
    - outlier_indices (numpy.ndarray): Indices of outlier values in the original array.
    """
    # Calculate mean and standard deviation
    mean = np.mean(arr)
    std_dev = np.std(arr)

    # Identify outliers
    outliers = arr[np.abs((arr - mean) / std_dev) > threshold]
    outlier_indices = np.where(np.abs((arr - mean) / std_dev) > threshold)[0]

    return outliers, outlier_indices


def extract_column_for_obj(data, obj_id, column):
    """
    Extract values for a specific column for a particular object across frames from good_data.
    
    Args:
    - data (dict): Filtered data with structure:
                        {frame_number: {obj_id: {column_name: value, ...}, ...}, ...}
    - obj_id (int): The ID of the object to extract column values for.
    - column (str): The column name to extract values for.
    
    Returns:
    - values (list): List of values for the target object and column across frames.
    """
    values = []
    
    # Traverse through good_data to collect values for the obj_id and column_name
    for frame, objects in data.items():
        if obj_id in objects:
            value = objects[obj_id].get(column, None)
            if value is not None:
                values.append(value)
    
    return np.array(values)


def get_aggregated_data(good_data, bad_data, fps=30, scale_factor=0.05):
    # calculate aggregated data
    aggregated_data = {}

    first_frame = list(good_data.keys())[0]
    obj_ids = extract_obj_ids(good_data)

    for obj_id in obj_ids:
        temp_data = {}

        # calculate min, max, mean and standard deviation of sizes
        sizes = extract_column_for_obj(good_data, obj_id, 'polygon size (mm2)')
        mean_size = np.mean(sizes)
        temp_data['mean polygon size (mm2)'] = mean_size

        # calculate 'is elongated'
        is_elongated = extract_column_for_obj(good_data, obj_id, 'is elongated')

        # calculate mean speeds elongated and overall
        speeds = extract_column_for_obj(good_data, obj_id, 'speed (mm/second)')
        mean_speed_elongated = np.mean(speeds * is_elongated)
        mean_speed = np.mean(speeds)
        temp_data['mean speed in elongated state (mm/second)'] = mean_speed_elongated
        temp_data['mean speed (mm/second)'] = mean_speed

        # calculate during 'elongated' state
        distances = extract_column_for_obj(good_data, obj_id, 'distance (mm)')
        temp_data['distance during elongated state (mm)'] = np.sum(distances * is_elongated)
        temp_data['distance (mm)'] = np.sum(distances)

        # calculate duration in elongated state
        temp_data['duration in elongated state (seconds)'] = np.sum(is_elongated) / fps

        # calculate duration in bent state
        temp_data['duration in bent state (seconds)'] = np.sum(1 - is_elongated) / fps

        good_frames = extract_column_for_obj(good_data, obj_id, 'frame')
        bad_frames = extract_column_for_obj(bad_data, obj_id, 'frame')
        temp_data['number of good frames'] = len(good_frames)
        temp_data['number of problematic frames'] = len(bad_frames)
        temp_data['first problematic frame'] = None if len(bad_frames) == 0 else bad_frames[0]

        aggregated_data[obj_id] = temp_data

    return aggregated_data


def write_raw_data(out_dir, csv_dir, data, index_label, exclude=['line mask']):
    os.makedirs(os.path.join(out_dir, csv_dir), exist_ok=True)

    if not data:
        return
    
    first_frame = list(data.keys())[0]
    obj_ids = extract_obj_ids(data)
    keys = [
        'frame',
        'second',
        'centroid x',
        'centroid y',
        'polygon size (pixels)',
        'polygon size (mm2)',
        'ellipse major axis (pixels)',
        'ellipse minor axis (pixels)',
        'ellipse major/minor (ratio)',
        'ellipse angle (degrees)',
        'major/minor ratio z-score (based on defined threshold)',
        'is elongated',
        'distance (pixels)',
        'distance (mm)',
        'speed (pixels/frame)',
        'speed (mm/second)'
    ]

    for obj_id in obj_ids:
        data_to_write = {}
        
        frames = []
        # Traverse through good_data to find frames containing the target_obj_id
        for frame, objects in data.items():
            if obj_id in objects:
                frames.append(frame)
        for frame in frames:
            temp_data = {}
            for key in keys:
                if key not in exclude:
                    temp_data[key] = data[frame][obj_id][key]
            data_to_write[frame] = temp_data
            
        df = pd.DataFrame(data_to_write)
        df = df.T
        df.to_csv(os.path.join(out_dir, csv_dir, f'{obj_id}.csv'), index=False, float_format='%.4f')


def write_aggregated_data(out_dir, aggregated_data, fname):
    df = pd.DataFrame(aggregated_data)
    df = df.T
    df.to_csv(os.path.join(out_dir, fname), index=True, index_label='track', float_format='%.4f')


def draw_track(out_dir, paths_dir, video_segments, h, w):
    os.makedirs(os.path.join(out_dir, paths_dir), exist_ok=True)

    first_frame = list(video_segments.keys())[0]
    obj_ids = list(video_segments[first_frame].keys())

    for obj_id in obj_ids:
        x_list = []
        y_list = []
        elongated_list = []

        # Load the track data from CSV
        track_csv_path = os.path.join(out_dir, 'raw_frames', f'{obj_id}.csv')
        track_data = pd.read_csv(track_csv_path)

        for frame_index in range(len(video_segments)):
            x_list.append(video_segments[frame_index][obj_id]['centroid x'])
            y_list.append(video_segments[frame_index][obj_id]['centroid y'])
            elongated_list.append(track_data.iloc[frame_index]['is elongated'])

        canvas = np.ones((h, w, 3), np.int32) * 255

        pts = np.array(list(zip(x_list, y_list)), np.int32)

        # Define the color for the object
        color = colors(obj_id+4, True)

        def lighten_bgr(bgr, factor):
            # Ensure factor is between 0 and 1
            factor = max(0, min(factor, 1))
            # Calculate the lighter shade
            return tuple(int(c + (255 - c) * factor) for c in bgr)

        # Define a lighter shade of the color
        light_color = lighten_bgr(color, 0.7)

        for i in range(len(pts) - 1):
            if elongated_list[i] == 0:
                cv2.line(canvas, pts[i], pts[i+1], color=light_color, thickness=2)
            else:
                cv2.line(canvas, pts[i], pts[i+1], color=color, thickness=2)

        # Save the image
        cv2.imwrite(os.path.join(out_dir, paths_dir, f'{obj_id}.png'), canvas)


def sample_frames(num_frames, src_fps, target_fps):
    """Samples a subset of frames based on source and target FPS of a video
    """
    if target_fps == src_fps:
        return list(range(num_frames))

    step = src_fps / target_fps
    return sorted(set(map(int, np.arange(0, num_frames, step))))


def draw_on_video(video_path, out_dir, out_video_name, video_segments, fps, h, w, duration_of_tracking_path=10):
    """fps represents the FPS that all predictions (video segments) are generated at.
    """
    # Dictionary to store tracking history with default empty lists
    track_history = defaultdict(lambda: [])
    colors_cache = {}  # Cache colors to avoid recomputation

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    if fps > cap_fps:
        print(f'[WARNING]: Untested when FPS is greater than the original video.')

    # Initialize video writer to save the output video with the specified properties
    out_video_path = os.path.join(out_dir, out_video_name)
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (w, h))

    # Get total frames of video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get which frames should be sampled
    frame_indices_to_sample = sample_frames(frame_count, cap_fps, fps)

    # Index of the frame that has been sampled
    frame_index_target_fps = 0
    for frame_index in tqdm(range(frame_count), desc='Generating video'):
        # Read frame (automatically advances, so no need for `cap.set`)
        ret, frame = cap.read()

        if not ret:
            break

        # Only proceed if it's a frame that should be sampled
        if frame_index not in frame_indices_to_sample:
            continue

        # Create or reuse annotator for the frame
        annotator = Annotator(frame, line_width=2)

        results = video_segments.get(frame_index_target_fps, {})  # Get current frame results

        for track_id, data in results.items():
            if track_id not in colors_cache:
                colors_cache[track_id] = colors(track_id + 4, True)

            annotator.seg_bbox(mask=data['line mask'], mask_color=colors_cache[track_id], label=str(track_id))

            track = track_history[track_id]
            track.append((data['centroid x'], data['centroid y']))
            if len(track) > fps * duration_of_tracking_path:
                track.pop(0)
                
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))  # Avoid type conversion

            cv2.polylines(frame, [points], isClosed=False, color=colors_cache[track_id], thickness=2)

        # Write the annotated frame to the output video
        out.write(frame)

        # Increment the frame index at the target FPS (i.e. the data sampled)
        frame_index_target_fps += 1

    # Release the video writer and capture objects, and close all OpenCV windows
    out.release()
    cap.release()


# ##################################################################################################
#                                   Outlier Detection Code
# ##################################################################################################
def process_raw_data(
        data: dict,
        # Stage 1
        zero_run_length: int = 20,
        # Stage 2
        size_diff_thresh: int = 1, size_diff_infill_frames: int = 20,
        # Stage 3
        drift_signal_ksize: int = 201, drift_diff_ksize: int = 11, drift_min_len: int = 80,
        drift_dir_frac: int = 0.75, drift_median_signal_diff_thresh: int = 1,
        drift_infill_frames: int = 20,
        # Stage 4
        segment_median_diff_thresh: int = 1, baseline_segment_min_len: int = 200,
        # Stage 5
        size_std_thresh: int = 4, speed_std_thresh: int = 5, size_speed_infill_frames: int = 20,
        # Stage 6
        final_infill_frames: int = 20,
        # Overlaps
        overlap_gap_fill: int = 20, overlap_min_duration: int = 100,
        # Output
        out_dir: str | None = None
) -> tuple[dict, dict]:
    """Process raw data to detect outliers for each object across frames.

    Based on new 6x stage logic.
    Works by:
    1. Detects segments of 0 diff runs based on size. Flags these as outliers
    -- This detects when the polygon 'freezes'
    2. Breaks up remaining data into 'inlier' segments. Then, for each inlier segment:
        a. Identifies areas in the segment as outliers for frames where size diff > threshold
        b. Fills in gaps based on a threshold of frames. (i.e. setting extra outliers)
        c. Combines these frames with the overall outliers
    -- This detects when the polygon changes size too quickly
    3. Detects segments where there is a continually sustained drift in size gradient.
    -- This detects when an ID is lost and the size 'drifts'
    4. Again, breaks up remaining data into 'inlier' segments. Then considering all segments:
        a. Computes statistics on that segment (e.g. mean/median/size/std)
        b. Compares statistics across segments, and flags an entire segment as an outlier if it
           deviates too much
        c. Combines the outlier segments with the overall outliers
    -- This helps remove different 'segments', where the polygon may jump to a new object
    5. Apply original size/speed std logic to remaining inlier sections
    6. Perform a final gap fill to join up any non-contiguous sections. Very useful for joining
        neighbouring disjoint segments.

    Following this per-object, it will also identify overlapping objects and flag extra outlier
    frames so that for a detected overlapping section, only data for one larvae is retained.

    Args:
        data: Nested dictionary with structure: {frame_number: {obj_id: {'speed (mm/second)': value, ...}, ...}, ...}
        zero_run_length: Threshold for detecting a 0-diff run.
        size_diff_thresh: Threshold for detecting outliers based on absolute size difference value.
        size_diff_infill_frames: How many frames to infill for size differences.
        drift_signal_ksize: When detecting drifts in gradients, the kernel size of the median
            filter that will be applied to the size data.
        drift_diff_ksize: The median filter kernel size applied to the diff signal computed on the
            smoothed size data.
        drift_min_len: The minimum continual length of a gradient diff to be considered a run.
        drift_dir_frac: The fraction of datapoints in the run that need to match the overall diff
            direction. This controls slight noise leftover in the signal.
        drift_median_signal_diff_thresh: For extracted drift runs, the minimum difference in
            median-filtered signal values from the start to end of the run to be counted as a true
            drift.
        drift_infill_frames: Following extracting drift indices, the amount of infill that occurs.
        segment_median_diff_thresh: For each segment, flag it as an outlier if it differs from the
            median of the first segment by above this amount. This is an absolute threshold.
        baseline_segment_min_len: Instead of always choosing first segment, only do so if it has at
            least this many data points. If not, choose the next closest segment.
        size_std_thresh: Standard deviation threshold on size data to be considered outliers.
        speed_std_thresh: Standard deviation threshold on speed data to be considered outliers.
        size_speed_infill_frames: How many frames to infill based on size/speed std outliers.
        final_infill_frames: How many frames to infill right at the end. This can help to join
            segments.
        overlap_gap_fill: Gap fill for detecting overlapping tracks.
        overlap_min_duration: Minimum duration for two tracks to overlap to be deemed overlapping.
        out_dir: If given, folder to store output visualisations showing problematic frames
            per-object.

    Returns:
        good_data (dict): Data without the outliers.
        bad_data (dict): Data containing only the outliers.
    """
    # Initialize containers for good and bad data
    good_data = {}
    bad_data = {}

    # Extract speeds and sizes for each obj_id across frames
    obj_measures = {}
    for frame, objects in data.items():
        for obj_id, obj_data in objects.items():
            speed = obj_data.get('speed (mm/second)', None)
            size = obj_data.get('polygon size (mm2)', None)
            centroid_x = obj_data.get('centroid x', None)
            centroid_y = obj_data.get('centroid y', None)
            if speed is not None and size is not None:
                obj_measures.setdefault(obj_id, []).append((frame, speed, size, centroid_x, centroid_y))

    # Set store for outlier (bad) frame indices per-object
    bad_object_frame_indices = {}

    # Detect outliers for each object
    for obj_id, obj_measure in obj_measures.items():
        # Extract data for this object
        frames, speed_data, size_data, _, _ = zip(*obj_measure)  # Separate frames, speeds, and sizes
        speed_data, size_data = np.array(speed_data), np.array(size_data)

        # Store for ALL detected outlier indices
        composite_outlier_indices = np.array([], dtype=int)

        # Compute diff arrays for size/speed
        size_diff_data = np.diff(size_data, prepend=size_data[0] - 0.1)
        speed_diff_data = np.diff(speed_data, prepend=speed_data[0] - 0.1)

        # Extract max index (used for contiguous segments)
        max_index = len(size_data) - 1

        # 1. Detect 0-run segments and combine with ALL outlier indices
        zero_diff_size_outliers = detect_zero_diff_run(size_data, run_length=zero_run_length)
        composite_outlier_indices = np.union1d(composite_outlier_indices, zero_diff_size_outliers)
        composite_outlier_indices.sort()

        # Get the set of inlier segments
        inlier_segments = get_inverse_contiguous_segments(composite_outlier_indices, max_idx=max_index)

        # 2. Do size diff detection using inlier segments
        for seg_idx, inlier_segment in enumerate(inlier_segments):
            # Extract start/end indices of segment and set of indices
            seg_start_idx, seg_end_idx = inlier_segment
            inlier_indices = contiguous_segments_to_indices([inlier_segment])

            # Get size difference data for this segment
            seg_size_diff_data = size_diff_data[inlier_indices]

            # a. Detect outliers in this segment (indices are relative to segment)
            seg_outlier_indices = detect_value_threshold(
                np.abs(seg_size_diff_data), threshold=size_diff_thresh)

            # b. Fill in gaps
            seg_outlier_indices = fill_gaps_in_array(
                seg_outlier_indices, max_gap=size_diff_infill_frames, min_idx=seg_start_idx,
                max_idx=seg_end_idx)

            # c. Combine these indices with the composite set
            # First change them from relative to absolute
            seg_outlier_indices_absolute = seg_outlier_indices + seg_start_idx
            composite_outlier_indices = np.union1d(composite_outlier_indices, seg_outlier_indices_absolute)
        composite_outlier_indices.sort()

        # 3. Detect segments of sustained gradient drift
        drift_indices, smoothed_size_data, smoothed_size_smoothed_diff_data = detect_mostly_sustained_gradient_run(
            size_data, signal_smooth_kernel_size=drift_signal_ksize,
            delta_smooth_kernel_size=drift_diff_ksize, min_drift_length=drift_min_len,
            drift_fraction=drift_dir_frac, median_run_diff_thresh=drift_median_signal_diff_thresh)

        # Fill gaps in drift_indices
        drift_indices = fill_gaps_in_array(
            drift_indices, max_gap=drift_infill_frames, max_idx=max_index)

        # Join these outliers with the rest of the data
        composite_outlier_indices = np.union1d(composite_outlier_indices, drift_indices)
        composite_outlier_indices.sort()
        inlier_segments = get_inverse_contiguous_segments(composite_outlier_indices, max_idx=max_index)

        # 4. Look across inlier segments and detect outliers
        if len(inlier_segments) > 1:
            # Get the raw 'size' data values for each inlier segment
            inlier_segment_sizes = []
            for inlier_segment in inlier_segments:
                inlier_segment_indices = contiguous_segments_to_indices([inlier_segment])
                inlier_segment_sizes.append(size_data[inlier_segment_indices])

            outlier_segment_indexes, medians, baseline_median_idx, upper_thresh, lower_thresh = detect_outlier_segment_median_diff(
                inlier_segment_sizes, median_diff_thresh=segment_median_diff_thresh,
                min_baseline_segment_len=baseline_segment_min_len)
            outlier_indices = []
            for outlier_seg_idx in outlier_segment_indexes:
                outlier_segment = inlier_segments[outlier_seg_idx]
                outlier_segment_indices = contiguous_segments_to_indices([outlier_segment])
                outlier_indices.extend(outlier_segment_indices)
            outlier_indices = np.array(outlier_indices, dtype=int)
            outlier_indices.sort()

            # Join these outliers with the rest of the data
            composite_outlier_indices = np.union1d(composite_outlier_indices, outlier_indices)
        composite_outlier_indices.sort()

        # 5. Apply original logic for detecting outliers based on size/speed mean/std values
        # Get indices of all remaining inlier points
        inlier_segments = get_inverse_contiguous_segments(composite_outlier_indices, max_idx=max_index)
        inlier_indices = contiguous_segments_to_indices(inlier_segments)

        # Get size and speed outliers (Returns indices into inlier_indices)
        inlier_size_data, inlier_speed_data = size_data[inlier_indices], speed_data[inlier_indices]
        _, inlier_size_outlier_indices, used_size_mean, used_size_std = detect_std_outliers(
            inlier_size_data, size_std_thresh)
        _, inlier_speed_outlier_indices, used_speed_mean, used_speed_std = detect_std_outliers(
            inlier_speed_data, speed_std_thresh)

        # Extract absolute indices from inlier_indices-relative
        abs_size_outlier_indices = inlier_indices[inlier_size_outlier_indices]
        abs_speed_outlier_indices = inlier_indices[inlier_speed_outlier_indices]
        abs_size_speed_outlier_indices = np.union1d(abs_size_outlier_indices, abs_speed_outlier_indices)

        # Fill in gaps in combined size/speed outlier
        abs_size_speed_outlier_indices = fill_gaps_in_array(
            abs_size_speed_outlier_indices, max_gap=size_speed_infill_frames, max_idx=max_index)

        # Add to the set of composite outlier frames
        composite_outlier_indices = np.union1d(composite_outlier_indices, abs_size_speed_outlier_indices)
        composite_outlier_indices.sort()

        # 6. Perform a final infill (to join up disjoint segments)
        composite_outlier_indices = fill_gaps_in_array(
            composite_outlier_indices, max_gap=final_infill_frames, max_idx=max_index)

        # Store these bad frame indices for the object
        bad_object_frame_indices[obj_id] = composite_outlier_indices

    # Detect overlapping areas for each object
    all_centroid_data, all_obj_ids = [], []
    for obj_id, obj_measure in obj_measures.items():
        _, _, _, centroid_x, centroid_y = zip(*obj_measure)
        centroid = [(x, y) for x, y in zip(centroid_x, centroid_y)]
        all_centroid_data.append(centroid)
        all_obj_ids.append(obj_id)
    overlaps = detect_overlapping_data(
        all_centroid_data, small_gap_fill=overlap_gap_fill,
        min_overlap_duration=overlap_min_duration)
    # Update bad_object_frame_indices based on detected overlaps
    for overlap_info in overlaps:
        # Extract info about this overlap
        idx_a, idx_b, overlap_segments = overlap_info
        obj_id_a, obj_id_b = all_obj_ids[idx_a], all_obj_ids[idx_b]

        # For each overlap segment, determine which is the likely 'correct'/'outlier' one
        # Flag frames as outliers for the object with the most intersecting outlier frames
        for overlap_segment in overlap_segments:
            # Find intersections for each object with outliers and overlap
            current_overlap_indices = contiguous_segments_to_indices([overlap_segment])
            overlap_a_outlier_indices = get_indices_intersection(
                bad_object_frame_indices[obj_id_a],
                current_overlap_indices,
            )
            overlap_b_outlier_indices = get_indices_intersection(
                bad_object_frame_indices[obj_id_b],
                current_overlap_indices,
            )

            # Set this segment as an outlier for the case with more overlap frames as outliers
            if len(overlap_a_outlier_indices) < len(overlap_b_outlier_indices):
                bad_object_frame_indices[obj_id_b] = np.union1d(bad_object_frame_indices[obj_id_b], current_overlap_indices)
            else:
                bad_object_frame_indices[obj_id_a] = np.union1d(bad_object_frame_indices[obj_id_a], current_overlap_indices)

    # Visualise the outliers
    for obj_id, obj_measure in obj_measures.items():
        frames, speed_data, size_data, _, _ = zip(*obj_measure)
        size_diff_data = np.diff(size_data, prepend=size_data[0] - 0.1)
        speed_diff_data = np.diff(speed_data, prepend=speed_data[0] - 0.1)

        # Get outlier indices for this object
        object_outlier_frame_indices = bad_object_frame_indices[obj_id]

        os.makedirs(out_dir, exist_ok=True)
        outlier_segments = get_contiguous_segments(object_outlier_frame_indices)
        v_lines = get_flat_segment_start_end_indices(outlier_segments)
        plot_multiple_data([
            dict(y=size_data, title=f'Size w/ Outliers', x_label='Frame', y_label='Size (mm^2)',
                 point_idxs=object_outlier_frame_indices, v_lines=v_lines, v_spans=outlier_segments,
                 v_lines_spans_colour='red', scatter_size=1, point_idx_size=2, bottom_y_lim=0),
            dict(y=speed_data, title=f'Speed w/ Outliers', x_label='Frame', y_label='Speed (mm/s)',
                 point_idxs=object_outlier_frame_indices, v_lines=v_lines, v_spans=outlier_segments,
                 v_lines_spans_colour='red', scatter_size=1, point_idx_size=2, bottom_y_lim=0),
            dict(y=size_diff_data, title=f'Size Diff w/ Outliers', x_label='Frame', y_label='Size (mm^s)',
                 point_idxs=object_outlier_frame_indices, v_lines=v_lines, v_spans=outlier_segments,
                 v_lines_spans_colour='red', scatter_size=1, point_idx_size=2, bottom_y_lim=None),
            dict(y=speed_diff_data, title=f'Speed Diff w/ Outliers', x_label='Frame', y_label='Speed (mm/s)',
                 point_idxs=object_outlier_frame_indices, v_lines=v_lines, v_spans=outlier_segments,
                 v_lines_spans_colour='red', scatter_size=1, point_idx_size=2, bottom_y_lim=None),
        ], fig_title=f'Outliers. {len(object_outlier_frame_indices)} frames.',
        sub_title=f'{len(object_outlier_frame_indices)} outlier frames. {len(outlier_segments)} segments.',
        out_file=os.path.join(out_dir, f'{obj_id}.jpg'))

    # Store final set of good/bad frames per-object
    for obj_id, obj_measure in obj_measures.items():
        frames, _, speed_data, _, _ = zip(*obj_measure)

        # Get outlier indices for this object
        object_outlier_frame_indices = bad_object_frame_indices[obj_id]

        # Separate good and bad data
        good_frames = [frames[i] for i in range(len(speed_data)) if i not in object_outlier_frame_indices]
        bad_frames = [frames[i] for i in object_outlier_frame_indices]

        # Populate good_data
        for frame in good_frames:
            good_data.setdefault(frame, {}).setdefault(obj_id, {}).update(data[frame][obj_id])

        # Populate bad_data
        for frame in bad_frames:
            bad_data.setdefault(frame, {}).setdefault(obj_id, {}).update(data[frame][obj_id])

    return good_data, bad_data


# #############################
# Outlier Detection Algorithms
# #############################
def detect_value_run(data, value=0, run_length=20):
    """Detects a run of a particular value in a set of data for a given run length.

    Where a run of the value (of the correct length) is detected, starting from the first until last
    value, those data indices are returned.

    By default the value is 0 (i.e. detecting 0 runs).
    """
    # Convert data to numpy array
    np_data = np.array(data)

    # Detect where data is equal to value
    value_data = np_data == value

    # Work out where data changes from 0 -> 1 or 1 -> 0 (i.e. shifting to/from value)
    diff_locations = np.diff(value_data.astype(int))

    # Work out direction of change in signal (i.e. starts: 1 -> 0, ends: 0 -> 1)
    starts = np.where(diff_locations == 1)[0] + 1       # Add 1 to shift diff
    ends = np.where(diff_locations == -1)[0]

    # Handle edge cases (signal starts/ends with value)
    if value_data[0]:
        starts = np.insert(starts, 0, 0)    # prepend index 0
    if value_data[-1]:
        ends = np.append(ends, len(np_data) - 1)        # append last index

    # Filter runs by length
    long_runs = [(s, e) for s, e in zip(starts, ends) if (e - s + 1) >= run_length]

    # Collect all indices in a flat array
    if long_runs:
        indices = np.concatenate([np.arange(s, e + 1) for s, e in long_runs])
        return indices
    else:
        return np.array([], dtype=int)


def detect_zero_diff_run(data, run_length=20):
    """Detects a difference of 0 in a signal, returning idxs where 0 diffs are detected for a run.

    For a given position in the array, the diff is computed by that value subtracted by the
    previous value.

    The first position in the array is set to a diff of 1.
    """
    # Convert data to numpy array
    np_data = np.array(data)

    # Find the diff of the data signal, ensuring the first diff is 1
    diff_data = np.diff(np_data, prepend=np_data[0] - 1)

    # Find a run of 0's (0 diff) for the run length
    return detect_value_run(diff_data, value=0, run_length=run_length)


def detect_value_threshold(data, threshold):
    """Very simple outlier detection that returns the index when the value is above a threshold.
    """
    data = np.array(data)
    return np.where(data > threshold)[0]


def detect_mostly_sustained_gradient_run(
        data, signal_smooth_kernel_size=201, delta_smooth_kernel_size=11,
        min_drift_length=80, drift_fraction=0.75, median_run_diff_thresh=1,
):
    """Returns contiguous runs where there is a sustained directional gradient drift.

    First filters the signal with a median filter. Then computes the difference of that signal.
    Then smooths the difference signal a little.

    Finally, looks through the smoothed difference signal and looks for same-directional runs
    (of at least min_drift_length long), where at least (>=) drift_fraction % of values in that
    range have the same directional gradient.

    Once the runs are detected, a final filtering process is performed to only claim runs as
    genuine drifts where the difference between the smoothed data values at the start/end of the
    run meet a threshold.

    Returns:
        outlier_indices: The indices corresponding to detected outlier segments.
        smoothed_data: The data following the median filter.
        delta_signal: The delta signal (following smoothing).
    """
    # Smooth the original signal
    smoothed_data = median_filter(data, size=signal_smooth_kernel_size)

    # Look at the difference signal (and smooth this a little)
    delta_signal = np.diff(smoothed_data, prepend=smoothed_data[0])
    delta_signal = median_filter(delta_signal, size=delta_smooth_kernel_size)

    # Store for the final set of 'drift' runs
    drift_runs, i, n = [], 0, len(delta_signal)
    while i < n:
        # Skip points with 0 gradient
        if delta_signal[i] == 0:
            i += 1
            continue

        # Determine run direction based on current point (1 if positive, -1 if negative)
        dirn = 1 if delta_signal[i] > 0 else -1

        # Attempt to make a run from this point
        run_start, run_end = i, i

        # Search for the run (extending the end) while the majority of points have same gradient direction
        while run_end + 1 < n:
            # Extract the window to 1x more than the current run_end
            window = delta_signal[run_start: run_end + 2]
            # Count how many values in this window match the current direction
            majority_count = np.sum((window > 0) if dirn == 1 else (window < 0))
            # Update the run_end if there are currently enough frames in the window with the same direction
            # If not, don't keep searching
            if majority_count / len(window) >= drift_fraction:
                run_end += 1
            else:
                break

        # Keep the detected run if it meets the minimum length criteria
        if (run_end - run_start + 1) >= min_drift_length:
            drift_runs.append((run_start, run_end))

        # Continue search from after this run (regardless of if it was too short or not)
        # TODO: Consider if above condition not true, setting i += 1
        i = run_end + 1

    # Filter the drift runs based on median difference at start/end
    filtered_drift_runs = []
    for start, end in drift_runs:
        if abs(smoothed_data[end] - smoothed_data[start]) >= median_run_diff_thresh:
            filtered_drift_runs.append((start, end))

    # Convert runs into indices
    drift_indices = contiguous_segments_to_indices(filtered_drift_runs)

    return drift_indices, smoothed_data, delta_signal


def detect_outlier_segment_median_diff(
        segments, median_diff_thresh=1, baseline_segment_idx=0, min_baseline_segment_len=200,
):
    """Detects entire segments that are likely to be outliers.

    Based on computing the median of each segment. For every segment, if it differs from the median
    of the "baseline_segment_idx" segment, greater than the median_diff_thresh, then it's flagged
    as an outlier.

    By default, the baseline_segment_idx is the FIRST segment.

    NOTE: When choosing this baseline segment. It is only used if it meets the
        min_baseline_segment_len. If not, then the next valid segment is used (unless none exist),
        in which case the requested segment will be used. Note: This doesn't circle around.

    Baseline segment selection:
        1. Prefer baseline_segment_idx if its length >= min_baseline_segment_len.
        2. Otherwise choose the next segment (index > baseline_segment_idx) whose length >= min_baseline_segment_len.
        3. If none exist, fall back to baseline_segment_idx.

    Segments should be a list of arrays, where each array contains the segment values.

    Returns indices of segments that are OUTLIERS.
    """
    # Compute median of each segment
    medians, lengths = [], []
    for segment in segments:
        medians.append(np.median(segment))
        lengths.append(len(segment))
    medians, lengths = np.array(medians), np.array(lengths)

    # Select the baseline median
    # Based on requested segment (if it meets the criteria), otherwise the next best one
    if lengths[baseline_segment_idx] >= min_baseline_segment_len:
        selected_baseline_idx = baseline_segment_idx
    else:
        valid_indices = np.where(lengths >= min_baseline_segment_len)[0]
        if len(valid_indices) > 0:
            selected_baseline_idx = valid_indices[0]
        else:
            # Fall back to the baseline if no better option found
            selected_baseline_idx = baseline_segment_idx

    # Extract the baseline median
    baseline_median = medians[selected_baseline_idx]

    # Compute deviations from the baseline
    deviations = np.abs(medians - baseline_median)

    # Work out which ones are outliers (differing by too much)
    outlier_segment_indices = np.where(deviations > median_diff_thresh)[0]

    # Identify upper/lower bounds for threshold
    upper_thresh = baseline_median + median_diff_thresh
    lower_thresh = baseline_median - median_diff_thresh

    return outlier_segment_indices, medians, selected_baseline_idx, upper_thresh, lower_thresh


def detect_std_outliers(data, std_threshold, mean=None, std=None):
    """Detects outliers in a Numpy array using mean and standard deviation.

    Data is first normalised before detecting outliers (e.g. (data - mean) / std)

    Values > +- std_threshold from the mean are flagged as outliers.

    Returns the indices into data corresponding to outliers.

    Args:
        data: Input array.
        std_threshold: Standard deviations from mean to consider a value an outlier.
        mean: Mean value of the dataset. If not given, is computed based on provided data.
        std: Standard deviation of the dataset. If not given, is computed based on provided data.

    Returns:
        outliers: Array of outlier values.
        outlier_indices: Indices of outlier values in the original array.
        mean: The mean value.
        std_dev: The standard deviation.
    """
    # Calculate mean and std if not given
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)

    # Normalise data
    data_norm = (data - mean) / std

    # Identify outliers
    outliers = data[np.abs(data_norm) > std_threshold]
    outlier_indices = np.where(np.abs(data_norm) > std_threshold)[0]

    return outliers, outlier_indices, mean, std


# #################
# Overlap Detection
# #################
def detect_overlapping_data(all_data, small_gap_fill=20, min_overlap_duration=100):
    """Detects whether any two sets of data overlap, and where that overlap occurs.

    Overlap only triggered if it meets the min_overlap_duration (e.g. for 100 indices).

    Expects all data to be a list of arrays/lists. The length of each sublist should be the same.

    Returns a list of tuples containing for any detected overlaps:
        (i, j, [(start_idx, end_idx), ...])
    Where:
        i: Is the index of the first track
        j: Is the index of the second track
        [(), ...]: Are the indices where overlaps were detected
    """
    # Ensure all data are numpy arrays
    all_data = [np.array(dat) for dat in all_data]

    # Store of final detected overlaps
    overlaps = []

    # Determine how many sets of data we have, and the length of each
    num_datasets = len(all_data)
    data_len = len(all_data[0])

    # Compare all pairs of tracks
    for i, j in combinations(range(num_datasets), 2):
        # i, j take on indices into all_data to compare
        dat_a, dat_b = all_data[i], all_data[j]

        # Ensure shapes match
        if dat_a.shape != dat_b.shape:
            raise RuntimeError(f'Data {i} and {j} have different shapes: {dat_a.shape} != {dat_b.shape}')

        # Create an array where points are equal. Work with (N,) or (N, x) dimensional arrays
        if dat_a.ndim == 1:
            eq = dat_a == dat_b
        else:
            eq = np.all(dat_a == dat_b, axis=1)

        # Find consecutive stretches of True
        if np.any(eq):
            # Pad eq with False at each end to detect overlaps at the start (Trues)
            padded = np.r_[False, eq, False]

            # Find where the array changes from False -> True/True -> False
            diff = np.diff(padded.astype(int))

            # Extract starts/ends of overlaps based on where signal changes sign
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            # Convert ends to be inclusive
            ends -= 1

            # Construct regions of (start, end)
            valid_overlap_segments = [(s, e) for s, e in zip(starts, ends)]

            # Fill gaps in this signal (to avoid slight noise)
            valid_overlap_indices = contiguous_segments_to_indices(valid_overlap_segments)
            valid_overlap_indices = fill_gaps_in_array(valid_overlap_indices, max_gap=small_gap_fill, max_idx=data_len-1)
            valid_overlap_segments = get_contiguous_segments(valid_overlap_indices)

            # Filter remaining segments based on meeting the length criteria
            valid_overlap_segments = [seg for seg in valid_overlap_segments if (seg[1] - seg[0] + 1) >= min_overlap_duration]
            if valid_overlap_segments:
                overlaps.append((i, j, valid_overlap_segments))
    return overlaps


# ###############
# Array Utilities
# ###############
def get_contiguous_segments(indices):
    """Returns the sequence of contiguous sections in an array.

    These are given as (start, end) pairs.
    """
    if len(indices) == 0:
        return []
    indices = np.sort(indices)

    # Find where gaps occur
    gaps = np.where(np.diff(indices) > 1)[0]

    # From these gaps, work out the starts/ends
    starts = np.insert(indices[gaps + 1], 0, indices[0])
    ends = np.append(indices[gaps], indices[-1])

    # Return the list of pairs of start/ends
    return list(zip(starts, ends))


def get_inverse_contiguous_segments(indices, max_idx, min_idx=0):
    """Returns the inverse sequence of contiguous sections in an array.

    These are given as (start, end) pairs.

    min/max idxs used to know what to pad out to

    If no indices in array, returns sequence of min -> max idx
    """
    if len(indices) == 0:
        return [(min_idx, max_idx)]
    indices = np.sort(indices)

    # First get the positive (present) sections
    pos_sections = get_contiguous_segments(indices)

    # Create store for inverse sections
    inv = []

    # Handle the gap before the first present section
    if pos_sections[0][0] > min_idx:
        inv.append((min_idx, pos_sections[0][0] - 1))

    # Handle gaps between present sections
    for (s1, e1), (s2, e2) in zip(pos_sections[:-1], pos_sections[1:]):
        if s2 > e1 + 1:
            inv.append((e1 + 1, s2 - 1))

    # Handle the gap after the last present section
    if pos_sections[-1][1] < max_idx:
        inv.append((pos_sections[-1][1] + 1, max_idx))

    # Return the list of pairs of start/ends
    return inv


def contiguous_segments_to_indices(segments):
    """Converts a set of contiguous segments to an array of indices.
    """
    out = []
    for s, e in segments:
        out.extend(range(s, e + 1))
    return np.array(out, dtype=int)


def get_indices_intersection(indices_a, indices_b):
    """Returns a list of indices that intersect both A and B.

    Done by finding the intersection between both.
    """
    return np.intersect1d(indices_a, indices_b)


def fill_gaps_in_array(indices, max_gap=2, min_idx=0, max_idx=None):
    """Fills in gaps in an array of indices.

    Values are filled in up to max_gap.

    i.e. if array was:
    [3430, 3431, 3432, 3435, 3450]
    would produce:
    [3430, 3431, 3432, 3433, 3434, 3435, 3450]

    If min_idx/max_idx given, will also pad out to the edges (i.e. to min_idx and max_idx)
    """
    if len(indices) == 0:
        return indices

    # First ensure sorted order
    indices = np.sort(indices)

    # Array of final 'filled' values
    filled = [indices[0]]

    # Look at pairs of values (first to second)
    for a, b in zip(indices[:-1], indices[1:]):
        # If there is a difference, and that difference is more than max_gap apart
        gap = b - a
        if 1 < gap <= max_gap + 1:
            # Fill in the values (from a + 1 to just before b)
            filled.extend(range(a + 1, b))
        # Then add b to the array
        filled.append(b)

    # Apply min/max edge padding
    start, end = filled[0], filled[-1]

    # Pad to min_idx
    if min_idx is not None and start - min_idx <= max_gap:
        filled = list(range(min_idx, start)) + filled

    # Pad to max_idx
    if max_idx is not None and max_idx - end <= max_gap:
        filled.extend(range(end + 1, max_idx + 1))

    # Filled in, so return as a new array
    return np.array(filled, dtype=int)


def get_flat_segment_start_end_indices(segments):
    """Returns a 1D list of all segment start/end indices.

    Removes duplicate values (i.e. may happen for segments of size 1).
    """
    return sorted(set([x for a, b in segments for x in (a, b)]))


# ##################################################################################################
#                                       Plotting Helpers
# ##################################################################################################
def plot_multiple_data(
        data_config, fig_title=None, sub_title=None, fig_size=(19.2, 10.8), blocking=True,
        out_file=None, n_cols=None,
):
    """Plots multiple datasets on different subplots of the same figure.

    Args:
        data_config: A list of dictionaries. Each dictionary containing keys matching what
            plot_data expects.
        fig_title: Title to give the figure
        sub_title: Title to show on the figure
        fig_size: Size of the plot
        blocking: Whether the plot should halt the program until closed, or render in the
            background.
        out_file: Filepath to store an image of the figure to. If specified, won't show a GUI.
        n_cols: If given, sets a specific number of columns. Otherwise is automatically computed.
    """
    # Determine how many plots to create
    num_plots = len(data_config)
    if n_cols is None:
        n_cols = math.ceil(math.sqrt(num_plots))
    n_rows = math.ceil(num_plots / n_cols)

    # Set up plotting area
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axes = axes.flatten()

    # Create data for that axis
    for idx, plot_config in enumerate(data_config):
        ax = axes[idx]
        plot_data(**plot_config, ax=ax)

    # Set title of whole figure (if given)
    if fig_title is not None:
        fig.canvas.manager.set_window_title(fig_title)

    # Other plot configuration
    plt.tight_layout()

    # Set title on the figure
    if sub_title is not None:
        fig.suptitle(sub_title, color='red')
        fig.subplots_adjust(top=0.95)

    # Show/save the plot
    if out_file is None:
        plt.show(block=blocking)
    else:
        plt.savefig(out_file)
        plt.close()


def plot_data(
        # Main data to plot
        y, x=None, plot_type='scatter',

        # Supplementary data
        point_idxs=None,
        h_lines=None, h_lines_labels=None, h_lines_ranges=None,
        v_lines=None, v_lines_labels=None,
        v_spans=None, v_lines_spans_colour='green',

        # Plot configuration
        title=None, x_label=None, y_label=None, fig_size=(19.2, 10.8),
        scatter_size=None, point_idx_size=None, bottom_y_lim=0,

        # Where to plot data (i.e. existing axes or create new plot)
        ax=None,
):
    """Plots a series of data on a scatter/line plot.

    if x not given, x-values set based on indices of y-values.

    Args:
        y: The series of y coordinates to plot.
        x: The series of corresponding x coordinates to plot. If not given, plots x-axis based on
            indices of each y coordinate.
        plot_type: The type of plot to generate. Options: scatter, line, bar.
        point_idxs: A set of indices (into y/x) that should be highlighted with a dot. The dot
            colour will be red. Supports a list of length 2, where colours will be red and green.
            Green dots will be drawn first.
        h_lines: Any y-axis coordinates that a horizontal line should be plotted at. Can be a list.
        h_lines_labels: Text to prepend hline labelling.
        h_lines_ranges: The x-axis range to draw the h_line. By default, draws across the entire
            x-axis. These should be specific x-values to draw from/to. Should be 1x per-h_line. If
            a specific one is None, draws a h_line across the whole axis.
        v_lines: Any x-axis coordinates that a vertical line should be plotted at. Can be a list.
        v_lines_labels: Text to prepend vline labelling.
        v_spans: Any (x1, x2) coordinates that a vertical span should be plotted at. Can be a list.
        v_lines_spans_colour: Colour to use for v_lines and v_spans.
        title: Title for the plot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        fig_size: Size for the figure.
        scatter_size: When plot_type is scatter, the size of the points to draw.
        point_idx_size: When drawing point_idxs, the size of those points to draw.
        bottom_y_lim: The bottom y-axis limit to set. If None, uses the default behaviour.
        ax: The axis to create data on. If None, a new plot is created. If given, data drawn to
            that axis.
    """
    create_new_plot = ax is None

    # Set up plot
    if create_new_plot:
        fig, ax = plt.subplots(figsize=fig_size)

    # Set up x data if not given
    if x is None:
        x = [i for i in range(len(y))]

    # Plot data
    if plot_type == 'scatter':
        ax.scatter(x, y, s=scatter_size)
    elif plot_type == 'line':
        ax.plot(x, y)
    elif plot_type == 'bar':
        ax.bar(x, height=y)
    else:
        raise NotImplementedError(f'Plot type {plot_type} not supported.')

    # Highlight specific points if required
    if point_idxs is not None:
        # Check if given two sets of point indexes
        if isinstance(point_idxs, list) and len(point_idxs) == 2 and not isinstance(point_idxs[0], int):
            for p_idxs, colour in zip(point_idxs[::-1], ('green', 'red')):
                x_pts = [x[idx] for idx in p_idxs]
                y_pts = [y[idx] for idx in p_idxs]
                ax.scatter(x_pts, y_pts, c=colour, s=point_idx_size)
        else:
            x_pts = [x[idx] for idx in point_idxs]
            y_pts = [y[idx] for idx in point_idxs]
            ax.scatter(x_pts, y_pts, c='red', s=point_idx_size)

    # Plot horizontal lines if required
    if h_lines is not None:
        if not isinstance(h_lines, list):
            h_lines = [h_lines]
        if h_lines_labels is not None:
            if not isinstance(h_lines_labels, list):
                h_lines_labels = [h_lines_labels]
            if len(h_lines) != len(h_lines_labels):
                raise RuntimeError(f'Mismatch in h_lines and labels. '
                                   f'{len(h_lines)} != {len(h_lines_labels)}')
        if h_lines_ranges is not None:
            if len(h_lines) != len(h_lines_ranges):
                raise RuntimeError(f'Must have match of h_lines_ranges to h_lines. '
                                   f'{len(h_lines)} != {len(h_lines_ranges)}')
        for h_idx in range(len(h_lines)):
            h_line = h_lines[h_idx]
            h_line_text = f'{h_line:.2f}'
            if h_lines_labels is not None:
                h_line_text = f'{h_lines_labels[h_idx]}: {h_line_text}'
            h_line_text_x_offset = 0
            if h_lines_ranges is None or h_lines_ranges[h_idx] is None:
                # If not specified, draw across entire axis
                ax.axhline(y=h_line, c='orange')
            else:
                # Plot a line from the coordinates instead
                h_line_xmin, h_line_xmax = h_lines_ranges[h_idx]
                ax.plot([h_line_xmin, h_line_xmax], [h_line, h_line], color='orange')
                h_line_text_x_offset = h_line_xmin
            ax.text(h_line_text_x_offset + 0.1, h_line + 0.1, h_line_text, color='r', ha='left', va='bottom')

    # Plot vertical lines if required
    if v_lines is not None:
        if not isinstance(v_lines, list):
            v_lines = [v_lines]
        if v_lines_labels is not None:
            if not isinstance(v_lines_labels, list):
                v_lines_labels = [v_lines_labels]
            if len(v_lines) != len(v_lines_labels):
                raise RuntimeError(f'Mismatch in v_lines and labels. '
                                   f'{len(v_lines)} != {len(v_lines_labels)}')
        for v_idx in range(len(v_lines)):
            v_line = v_lines[v_idx]
            v_line_text = f'{v_line:d}'
            if v_lines_labels is not None:
                v_line_text = f'{v_lines_labels[v_idx]}: {v_line_text}'
            ax.axvline(x=v_line, c=v_lines_spans_colour)
            ax.text(
                v_line, np.nanmax(y) * 1.01, v_line_text, rotation=90, color=v_lines_spans_colour,
                ha='center', va='bottom')

    # Plot vertical spans if required
    if v_spans is not None:
        if not isinstance(v_spans, list):
            v_spans = [v_spans]
        for v_idx in range(len(v_spans)):
            v_start, v_end = v_spans[v_idx]
            ax.axvspan(v_start, v_end, alpha=0.25, color=v_lines_spans_colour)

    # Configure the plot
    if bottom_y_lim is not None:
        ax.set_ylim(bottom=bottom_y_lim)
    ax.grid()
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    # Show the plot
    if create_new_plot:
        plt.show()

