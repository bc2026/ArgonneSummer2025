import pandas as pd
import numpy as np
import re
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from OutlierUtils import *
from OutlierUtils import is_mergeable


def get_detected_intervals(shifts, first_region):

    detected_intervals = first_region

    for i in range(len(shifts) - 1):
        start_p = shifts.iloc[i]
        next_p = shifts.iloc[i+1]
        detected_intervals.append((start_p, next_p))
    return detected_intervals

def get_detected_regions(left_derivatives_df_c: pd.DataFrame, first_region_end_index: int, response: str):

    shifts = left_derivatives_df_c[left_derivatives_df_c['Left_Derivative'] > left_derivatives_df_c['Y']]

    # Shifts are defined where the f(x) > f'(x) after the starting region
    shifts = shifts.reset_index(drop=True)
    shifts = shifts['Time']

    detected_intervals = [(0, first_region_end_index), (first_region_end_index, shifts[0])]

    return detected_intervals, shifts


def get_region(df, t0, tf):
    mask = (df['Time'] >= t0) & (df['Time'] <= tf)
    return df[mask]


def calculate_2d_median(X1, X2):
    if X1.size == 0 or X2.size == 0:
        print("Warning: Input 'points' array is empty. Returning an empty array.")
        return np.array([])

    # Calculate the median of the first column (X coordinates)
    median_x1 = np.median(X1)

    # Calculate the median of the second column (Y coordinates)
    median_x2 = np.median(X2)

    # Return the 2D median as a NumPy array
    return np.array([median_x1, median_x2])


def merge_close_points(df: pd.DataFrame, regions: list, p: float, response: str):
    if not regions:
        return []

    merged_regions = []
    curr_region = regions[0]
    curr_region_df = get_region(df, curr_region[0], curr_region[1])
    curr_median = calculate_2d_median(curr_region_df['Time'], curr_region_df[response])

    for i in range(1, len(regions)):
        next_region = regions[i]
        next_region_df = get_region(df, next_region[0], next_region[1])
        next_median = calculate_2d_median(next_region_df['Time'], next_region_df[response])

        if not is_mergeable(curr_median, next_median, p):
            merged_regions.append(curr_region)
            curr_region = next_region
            curr_median = next_median
        else:
            # Extend curr_region to include next_region
            curr_region = (curr_region[0], next_region[1])
            curr_region_df = get_region(df, curr_region[0], curr_region[1])
            curr_median = calculate_2d_median(curr_region_df['Time'], curr_region_df[response])

    # Append the final merged region
    merged_regions.append(curr_region)
    return merged_regions


def sanitize_response_name(response):
    # Replace slashes with underscores
    clean = response.replace('/', '_')
    # Remove parentheses and commas
    clean = re.sub(r'[(),]', '', clean)
    # Replace spaces and dashes with underscores
    clean = re.sub(r'[\s\-]+', '_', clean)
    return clean.strip('_')  # remove leading/trailing underscores if any

