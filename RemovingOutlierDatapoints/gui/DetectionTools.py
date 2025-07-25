import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils.OutlierUtils import *
from utils.OutlierUtils import is_mergeable


def get_detected_intervals(shifts, first_region):
    """
    Get detected intervals from shifts
    """
    intervals = []
    
    if first_region:
        intervals.extend(first_region)
    
    for shift in shifts:
        intervals.append((shift, shift + 1))  # Simple interval around each shift
    
    return intervals


def get_detected_regions(left_derivatives_df_c: pd.DataFrame, first_region_end_index: int, response: str):
    """
    Get detected regions from left derivatives DataFrame
    """
    # Filter for shifts (points where left derivative > Y)
    shifts_df = left_derivatives_df_c[left_derivatives_df_c['Left_Derivative'] > left_derivatives_df_c['Y']]
    
    # Get regions based on shifts
    regions = []
    
    # Add first region
    regions.append((0, first_region_end_index))
    
    # Add regions based on shifts
    for shift_time in shifts_df.index:
        regions.append((shift_time, shift_time + 10))  # 10 unit intervals around shifts
    
    return regions


def get_region(df, t0, tf):
    """
    Get data points within a time range
    """
    mask = (df['Time'] >= t0) & (df['Time'] <= tf)
    return df[mask]


def calculate_2d_median(X1, X2):
    """
    Calculate 2D median of two arrays
    """
    if len(X1) == 0 or len(X2) == 0:
        print("Warning: Input arrays are empty. Returning empty array.")
        return np.array([])

    # Calculate the median of the first column (X coordinates)
    median_x1 = np.median(X1)

    # Calculate the median of the second column (Y coordinates)
    median_x2 = np.median(X2)

    # Return the 2D median as a NumPy array
    return np.array([median_x1, median_x2])


def merge_close_points(df: pd.DataFrame, regions: list, p: float, response: str):
    """
    Merges regions where the median of their response values are closer
    than a percentage threshold 'p'. 
    Also removes tiny regions that are likely transition artifacts.
    """
    if not regions:
        return []

    # Sort regions by start time to ensure proper order
    regions = sorted(regions, key=lambda x: x[0])
    
    # Calculate minimum region size (2% of total time range)
    if len(regions) > 0:
        total_time_range = regions[-1][1] - regions[0][0]
        min_region_size = total_time_range * 0.02  # Minimum 2% of total range
    else:
        min_region_size = 0
    
    # First pass: remove tiny regions by merging them with adjacent larger regions
    filtered_regions = []
    for i, region in enumerate(regions):
        region_size = region[1] - region[0]
        
        if region_size < min_region_size and len(filtered_regions) > 0:
            # Tiny region - merge with previous region
            prev_region = filtered_regions[-1]
            merged_region = (prev_region[0], region[1])
            filtered_regions[-1] = merged_region
            print(f"Merged tiny region {region} with previous region {prev_region} -> {merged_region}")
        else:
            filtered_regions.append(region)
    
    regions = filtered_regions
    merged_regions = []
    curr_region = regions[0]
    
    # Get the DataFrame for the current region
    curr_region_df = get_region(df, curr_region[0], curr_region[1])
    
    # Calculate the median of the response column for the current region
    if curr_region_df.empty:
        curr_median = np.nan
    else:
        curr_median = np.median(curr_region_df[response])

    print(f"Starting merge with {len(regions)} regions, p-value: {p}")
    print(f"First region: {curr_region}, median: {curr_median}")

    for i in range(1, len(regions)):
        next_region = regions[i]
        
        # Get the DataFrame for the next region
        next_region_df = get_region(df, next_region[0], next_region[1])
        
        # Calculate the median of the response column for the next region
        if next_region_df.empty:
            next_median = np.nan
        else:
            next_median = np.median(next_region_df[response])

        print(f"Comparing region {i}: {next_region}, median: {next_median}")

        # Handle cases where medians might be NaN
        if np.isnan(curr_median) or np.isnan(next_median):
            is_mergeable_result = False
            print(f"  -> Not mergeable (NaN median)")
        else:
            is_mergeable_result = is_mergeable(curr_median, next_median, p)
            print(f"  -> is_mergeable({curr_median:.2f}, {next_median:.2f}, {p}) = {is_mergeable_result}")

        # If NOT mergeable, add curr_region and move to next
        if not is_mergeable_result:
            merged_regions.append(curr_region)
            print(f"  -> Added region to merged list: {curr_region}")
            curr_region = next_region
            curr_median = next_median
        else:
            # If mergeable, extend curr_region to include next_region's span
            print(f"  -> Merging {next_region} into {curr_region}")
            curr_region = (curr_region[0], next_region[1])
            
            # Recalculate median for extended region
            extended_region_df = get_region(df, curr_region[0], curr_region[1])
            if not extended_region_df.empty:
                curr_median = np.median(extended_region_df[response])
            print(f"  -> Extended region: {curr_region}, new median: {curr_median}")

    # Add the final region
    merged_regions.append(curr_region)
    print(f"Final merged regions: {merged_regions}")
    
    return merged_regions


def sanitize_response_name(response):
    """
    Sanitize response name for file naming
    """
    # Replace slashes with underscores
    return response.replace('/', '_').replace('\\', '_').replace(' ', '_')

