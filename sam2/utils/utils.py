import copy
import os
from collections import defaultdict

# Set environment variable to disable ultralytics verbose logging
os.environ['YOLO_VERBOSE'] = "False"

import cv2
import ffmpeg
import numpy as np
import pandas as pd
import torch
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
    Convert frame numbers to data per second.
    
    Args:
        frame_dict (dict): Dictionary with frame numbers as keys.
        fps (int): Frames per second.
    
    Returns:
        dict: Dictionary with seconds as keys.
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
            
            if previous_frame_index < 0:
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


def process_raw_data(data, speed_outlier_std=5, size_outlier_std=4):
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


def draw_on_video(video_path, out_dir, out_video_name, video_segments, fps, h, w, duration_of_tracking_path=10):
    # Dictionary to store tracking history with default empty lists
    track_history = defaultdict(lambda: [])
    colors_cache = {}  # Cache colors to avoid recomputation

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize video writer to save the output video with the specified properties
    out_video_path = os.path.join(out_dir, out_video_name)
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (w, h))

    # Get total frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_index in tqdm(range(frame_count), desc='Generating video'):
        # Read frame (automatically advances, so no need for `cap.set`)
        ret, frame = cap.read()

        if not ret:
            break

        # Create or reuse annotator for the frame
        annotator = Annotator(frame, line_width=2)

        results = video_segments.get(frame_index, {})  # Get current frame results

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

    # Release the video writer and capture objects, and close all OpenCV windows
    out.release()
    cap.release()
