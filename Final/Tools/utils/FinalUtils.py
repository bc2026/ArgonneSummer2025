import logging

import pandas as pd
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import io


def load_ec_csv(path_to_file: str):
    """
    Loads electrochemical data from a fixed-width file.

    It finds the start of the data after a line containing "End Comments"
    and reads the three data columns (Voltage, Current, and Time).

    Args:
        path_to_file: The full path to the data file.

    Returns:
        A pandas DataFrame with a normalized time index.
    """
    # Step 1: Find the number of metadata rows to skip
    rows_to_skip = 0
    with open(path_to_file, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if "End Comments" in line:
                rows_to_skip = i + 1
                break
        else:
            raise ValueError("'End Comments' line not found in the file.")

    # Step 2: Define column names based on the metadata
    # The file has three data columns: Voltage, Current, and Time.
    column_names = ['Voltage (V)', 'Current (A/cm2)', 'Time (s)']

    # Step 3: Read the fixed-width file, skipping the header comments
    df = pd.read_fwf(
        path_to_file,
        skiprows=rows_to_skip,
        names=column_names
    )

    # Step 4: Normalize the time column and set it as the index
    if not df.empty:
        df['Time (s)'] = df['Time (s)'] - df['Time (s)'].min()

    df.set_index('Time (s)', inplace=True)
    df.sort_index(inplace=True)

    return df


def create_voltage_schedule(start_time: float, period: int, voltages: list, cycles: int) -> dict:
    """
    Creates a dictionary mapping timestamps to a repeating cycle of voltages.

    This function is useful for generating experimental protocols where a set of
    voltages needs to be applied sequentially and repeated over several cycles.

    Args:
        start_time (float): The initial timestamp for the schedule (e.g., in seconds).
        period (int): The time duration for each voltage step (e.g., in seconds).
        voltages (list): A list of voltage values to apply in one cycle.
        cycles (int): The number of times to repeat the voltage cycle.

    Returns:
        dict: A dictionary where keys are the calculated timestamps and values are
              the corresponding voltages from the specified sequence.
    """
    schedule = {}
    current_time = start_time

    # Create the full sequence of voltages by repeating the cycle
    full_voltage_sequence = voltages * cycles

    # Iterate through the full sequence and assign each voltage to a timestamp
    for voltage in full_voltage_sequence:
        schedule[current_time] = voltage
        current_time += period

    return schedule


def add_potential_to_dataframe(schedule: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a voltage to each time in a DataFrame based on a schedule.

    This function applies a 'step function' of voltage. For any time t in the
    DataFrame, the assigned voltage is the one from the most recent schedule
    entry before or at time t.

    Args:
        schedule (dict): A dictionary mapping start times (keys) to voltages (values).
        df (pd.DataFrame): The DataFrame to which the voltage column will be added.
                           It must have a time-based index.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'Applied_Voltage (V)' column.
    """

    logging.info(f'Input shape: {df.shape}, schedule length: {len(schedule)}')

    # Ensure both the DataFrame index and schedule are sorted by time
    df_processed = df.sort_index()
    schedule_series = pd.Series(schedule).sort_index()

    # Combine the DataFrame's index with the schedule's time points
    # This ensures that all voltage change-points are included
    combined_index = df_processed.index.union(schedule_series.index).sort_values()

    # Reindex the schedule to the combined index and forward-fill the values
    voltage_filled = schedule_series.reindex(combined_index, method='ffill')

    # Now, select only the values corresponding to the original DataFrame's index
    # This effectively maps the step-function values onto your original time points
    df_processed['Applied_Voltage (V)'] = voltage_filled.reindex(df_processed.index)

    logging.info(f'Output shape: {df_processed.shape}')

    return df_processed

def clean_white_space(df: pd.DataFrame, t: str):
    return df.rename(columns=lambda x: x.strip()).sort_values(t)


def load_trunc_icp_csv(icp_df: pd.DataFrame, del_start: int, response_delay: int) -> pd.DataFrame:
    del_start = int(del_start)
    response_delay = int(response_delay)

    print(f"[load_trunc_icp_csv] Starting with shape {icp_df.shape}, del_start={del_start}, response_delay={response_delay}")

    # Make a copy to avoid modifying the original DataFrame
    df = icp_df.copy()

    # Calculate time range before normalization for debugging
    time_min_orig = df.index.min()
    time_max_orig = df.index.max()
    time_range_orig = time_max_orig - time_min_orig
    print(f"[load_trunc_icp_csv] Original time range: {time_min_orig} to {time_max_orig} (range: {time_range_orig})")

    # IMPORTANT: Apply response_delay to raw timestamps BEFORE normalization
    # This ensures large initial timestamps (e.g., starting at 25,000) are properly handled
    if response_delay > 0:
        delay_point = time_min_orig + response_delay
        print(f"[load_trunc_icp_csv] Filtering raw data before normalization: t >= {delay_point}")
        mask = df.index >= delay_point
        rows_before = len(df)
        df = df[mask]
        rows_after = len(df)
        print(f"[load_trunc_icp_csv] Response delay filter removed {rows_before - rows_after} rows, {rows_after} rows remaining")

    # Now normalize time AFTER applying response_delay
    if not df.empty:
        df.index = df.index - df.index.min()
        print(f"[load_trunc_icp_csv] Normalized time range: {df.index.min()} to {df.index.max()}")
    else:
        print(f"[load_trunc_icp_csv] WARNING: No data left after response_delay filter!")
        return pd.DataFrame(columns=icp_df.columns)

    # Apply del_start if provided by removing first N points
    if del_start > 0:
        if len(df) <= del_start:
            print(f"[load_trunc_icp_csv] WARNING: del_start ({del_start}) is >= remaining rows ({len(df)})!")
            print(f"[load_trunc_icp_csv] This will result in an empty dataset. Consider using a smaller del_start value.")

        rows_before = len(df)
        df = df.iloc[del_start:]
        rows_after = len(df)
        print(f"[load_trunc_icp_csv] Removed first {del_start} points, {rows_after} rows remaining")

    # Check if we have any data left
    if len(df) == 0:
        print(f"[load_trunc_icp_csv] ERROR: No data remains after applying filters!")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=icp_df.columns)

    return df


def interpolate(ec_df: pd.DataFrame, icp_trunc_df: pd.DataFrame):
    """Interpolates ICP-MS and E-chem data"""
    try:
        print(f"[interpolate] Starting with ec_df shape: {ec_df.shape}, icp_trunc_df shape: {icp_trunc_df.shape}")

        # Convert index to numpy array, ensuring it's float type
        ec_times = ec_df.index.to_numpy(dtype=float)
        ec_potentials = ec_df['Voltage (V)'].values
        ec_density = ec_df.get('Current (A/cm2)', pd.Series(np.nan, index=ec_df.index)).values

        print(f"[interpolate] Extracted arrays: ec_times (len={len(ec_times)}), ec_potentials (len={len(ec_potentials)})")

        icp_times = icp_trunc_df.index
        print(f"[interpolate] Extracted icp_times (len={len(icp_times)})")

        # Check for valid data ranges before interpolation
        if len(ec_times) == 0 or len(icp_times) == 0:
            print("[interpolate] ERROR: Empty time arrays, cannot interpolate")
            # Return a copy of the input without interpolated data
            return icp_trunc_df.copy()

        # Ensure ec_times contains valid values
        if np.isnan(ec_times).any():
            print("[interpolate] WARNING: ec_times contains NaN values, cleaning...")
            valid_indices = ~np.isnan(ec_times)
            ec_times = ec_times[valid_indices]
            ec_potentials = ec_potentials[valid_indices] if len(ec_potentials) > 0 else ec_potentials
            ec_density = ec_density[valid_indices] if len(ec_density) > 0 else ec_density

        print("[interpolate] Performing interpolation...")
        # Perform interpolation, but only if we have valid data
        if len(ec_times) > 0 and len(ec_potentials) > 0:
            interp_potentials = np.interp(icp_times, ec_times, ec_potentials)
        else:
            print("[interpolate] WARNING: Cannot interpolate potentials, using NaN values")
            interp_potentials = np.full_like(icp_times, np.nan, dtype=float)

        if len(ec_times) > 0 and len(ec_density) > 0 and not np.all(np.isnan(ec_density)):
            # Remove NaN values from ec_density for interpolation
            valid_indices = ~np.isnan(ec_density)
            if np.any(valid_indices):
                interp_densities = np.interp(icp_times, ec_times[valid_indices], ec_density[valid_indices])
            else:
                interp_densities = np.full_like(icp_times, np.nan, dtype=float)
        else:
            print("[interpolate] WARNING: Cannot interpolate densities, using NaN values")
            interp_densities = np.full_like(icp_times, np.nan, dtype=float)

        print("[interpolate] Interpolation completed")

        fin_df = icp_trunc_df.copy()
        fin_df['Interp_Voltage (V)'] = interp_potentials
        fin_df['Interp_Current (A/cm2)'] = interp_densities

        # No longer saving CSV here since it's already done in backend.py
        print(f"[interpolate] Returning dataframe with shape {fin_df.shape}")
        return fin_df

    except Exception as e:
        print(f"[interpolate] ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())

        # Return original data as fallback to avoid hanging
        return icp_trunc_df.copy()