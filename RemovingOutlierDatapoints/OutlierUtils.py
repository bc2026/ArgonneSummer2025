import pandas as pd
import numpy as np
from collections import deque
from sklearn.metrics import auc
import re
import tkinter as tk
from tkinter import simpledialog, messagebox

def left_derivatives(X: np.array,Y: np.array) -> pd.DataFrame:
    left_derivs = []

    for i in range(1, len(X)):
        dx = X[i] - X[i-1]
        dy = Y[i] - Y[i-1]

        # Handle cases where dx is zero to avoid division by zero
        if dx == 0:
            left_derivs.append(np.inf if dy > 0 else (-np.inf if dy < 0 else np.nan))
        else:
            left_derivs.append(dy / dx)
    
    df = pd.DataFrame({
        'Time': X,
        'Y': Y,
        'Left_Derivative': left_derivs + [-1] 
    })

    return df




def merge_close_points(points, min_distance):
    """
    Merges points that are closer than min_distance.
    This is a placeholder function, replace with your actual implementation.
    A simple merging logic: keep only the first point if others are too close.
    """
    if not points:
        return []

    points = sorted(points)
    merged = [points[0]]
    for i in range(1, len(points)):
        if points[i] - merged[-1] >= min_distance:
            merged.append(points[i])
    return merged

def get_motive_points(X, Y, threshold_percentage, min_merge_distance=0):
    """
    Identifies motive points based on relative change in Y and merges
    points that are too close.

    Args:
        X (collections.deque): Deque of X-coordinates (e.g., time).
        Y (collections.deque): Deque of Y-coordinates (e.g., response values).
        threshold_percentage (float): The relative change threshold to identify a motive point.
        min_merge_distance (float): The minimum distance for merging motive points.
                                    Set to 0 to disable merging.

    Returns:
        list: A list of identified and merged motive points (X-coordinates).
    """
    if not X or not Y:
        return []

    # Ensure X and Y are deques
    X = deque(X)
    Y = deque(Y)

    motive_points = []
    if X: # Ensure X is not empty before popping
        curr_x, curr_y = X.popleft(), Y.popleft()
        motive_points.append(curr_x) # Always include the first point

        while X:
            nex_x, nex_y = X.popleft(), Y.popleft()

            if abs(curr_y) > 1e-8:  # avoid division by zero
                relative_change = abs(nex_y - curr_y) / abs(curr_y)
            else:
                relative_change = float('inf')  # treat as big change if curr_y ≈ 0

            if relative_change >= threshold_percentage:
                motive_points.append(nex_x) # Append nex_x, as it's the point where the change occurred

            curr_x, curr_y = nex_x, nex_y

    # Merge points if min_merge_distance is specified and greater than 0
    if min_merge_distance > 0:
        motive_points = merge_motive_points(motive_points, min_merge_distance)

    return motive_points


def sanitize_response_name(response):
    # Replace slashes with underscores
    clean = response.replace('/', '_')
    # Remove parentheses and commas
    clean = re.sub(r'[(),]', '', clean)
    # Replace spaces and dashes with underscores
    clean = re.sub(r'[\s\-]+', '_', clean)
    return clean.strip('_')  # remove leading/trailing underscores if any



def get_area(R=-1, data=pd.Series([])):
    if data.empty:
        return False
    
    mean = data.mean()
    data = data - mean

    if R is None:
        R = np.arrange(len(data))

    else:
        R = np.array(R)
    
    if len(R) != len(data):
        raise ValueError("The length of 'R' must match the length of 'data'.")

    sorted_indices = np.argsort(R)
    R_sorted = R[sorted_indices]
    data_sorted_by_R = data.iloc[sorted_indices]

    calculated_auc = auc(R_sorted, data_sorted_by_R)
    return calculated_auc


def remove_outliers_by_region(R: tuple, response_column_name='', Y=pd.DataFrame([])):
    """
    Removes outliers from a DataFrame based on a 3-standard deviation rule
    for a specified response column within a defined time region.

    Args:
        R (tuple): A tuple (start_time, end_time) defining the time region to consider.
        response_column_name (str): The name of the column in Y to analyze for outliers.
        Y (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed for the specified region. Returns an empty
                      DataFrame if the input Y is empty, the column doesn't exist, or if the column is not numeric.
    """

    # --- Input Validation ---
    # Check if the input DataFrame is empty or if the specified response column does not exist.
    if Y.empty or response_column_name not in Y.columns:
        print(f"Warning: DataFrame is empty or column '{response_column_name}' not found.")
        return pd.DataFrame([])

    # Ensure the response column is numeric, as outlier detection typically requires numeric data.
    if not pd.api.types.is_numeric_dtype(Y[response_column_name]):
        print(f"Warning: Column '{response_column_name}' is not numeric. Cannot detect outliers.")
        return Y.copy() # Return a copy as no changes were made

    # --- Time-based Filtering ---
    # Filter the DataFrame to include only rows within the specified time region [R[0], R[1]].
    # Assumes 'Time' is a column in the DataFrame Y.
    time_filtered_Y = Y[(Y['Time'] >= R[0]) & (Y['Time'] <= R[1])]



    # If, after time filtering, the DataFrame becomes empty, there's nothing to process.
    if time_filtered_Y.empty:
        print(f"Warning: No data found within the specified time region {R}.")
        return pd.DataFrame([])

    # --- Outlier Detection Calculation ---
    # Get the data for the response column from the time-filtered DataFrame.
    response_data = time_filtered_Y[response_column_name]
   
    # Calculate basic statistics for the response data within the region.
    # Y_bar: The mean of the response data.
    Y_bar = response_data.mean()
    # Y_std: The standard deviation of the response data, used for defining outlier bounds.
    Y_std = response_data.std()

    # Define 'k' for the k-standard deviation rule. Here, k=3 means values more than
    # 3 standard deviations away from the mean are considered outliers.
    k = 3

    # Calculate the lower bound (lb) for non-outliers.
    lb = Y_bar - k * Y_std

    # Calculate the upper bound (ub) for non-outliers.
    ub = Y_bar + k * Y_std

    # --- Outlier Removal ---
    # Create a boolean mask to identify non-outliers.
    # Rows where the response data is within [lb, ub] will be True, otherwise False.
    is_not_outlier = (response_data >= lb) & (response_data <= ub)

    # Filter the DataFrame based on the mask.
    Y_prime = time_filtered_Y[is_not_outlier].copy() # .copy() to avoid SettingWithCopyWarning

    return Y_prime


