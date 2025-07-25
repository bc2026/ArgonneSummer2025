import logging
import pandas as pd
import numpy as np
from collections import deque # Not used in provided functions, but kept if needed elsewhere
from sklearn.metrics import auc # Not used in provided functions, but kept if needed elsewhere
import re
import tkinter as tk # Not used in provided functions, but kept if needed elsewhere
from tkinter import simpledialog, messagebox # Not used in provided functions, but kept if needed elsewhere

# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def is_mergeable(X1, X2, p: float) -> bool:
    """
    Checks if two scalar values (X1 and X2) are mergeable based on a percentage difference.

    Args:
        X1: The first scalar value (e.g., median of response).
        X2: The second scalar value.
        p: The percentage threshold for merging (e.g., 5 for 5%).

    Returns:
        True if the absolute difference, relative to the absolute value of X1,
        is less than or equal to p/100; False otherwise.
    """
    # If the scalar values are identical, they are definitely mergeable
    if X1 == X2:
        logging.debug(f"is_mergeable({X1}, {X2}, {p}): True (identical values)")
        return True

    dist = abs(X2 - X1) # Absolute difference between the two scalar values

    norm_X1 = abs(X1) # Absolute value of the reference scalar X1

    # Handle the case where the reference scalar X1 is zero.
    # Use X2 as reference instead, or use a small epsilon if both are near zero
    if norm_X1 == 0:
        norm_X2 = abs(X2)
        if norm_X2 == 0:
            # Both are zero or very close to zero - they should be mergeable
            logging.debug(f"is_mergeable({X1}, {X2}, {p}): True (both near zero)")
            return True
        else:
            # Use X2 as reference when X1 is zero
            norm_reference = norm_X2
    else:
        norm_reference = norm_X1

    result = True if (dist / norm_reference <= p / 100) else False
    logging.debug(f"is_mergeable({X1}, {X2}, {p}): dist={dist}, norm_ref={norm_reference}, ratio={dist/norm_reference:.4f}, threshold={p/100:.4f} -> {result}")
    return result


def get_region(df: pd.DataFrame, t0, tf) -> pd.DataFrame:
    """
    Filters a DataFrame to include only rows within a specified time range.

    Args:
        df: The input pandas DataFrame.
        t0: The start time (inclusive).
        tf: The end time (inclusive).

    Returns:
        A new DataFrame containing rows where 'Time' is between t0 and tf.
    """
    mask = (df['Time'] >= t0) & (df['Time'] <= tf)
    filtered_df = df[mask]
    # Added print statement for debugging
    print(f"get_region called with t0={t0}, tf={tf}. Filtered DataFrame shape: {filtered_df.shape}")
    return filtered_df

def left_derivatives(X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    """
    Calculates the left derivatives for a series of X and Y points.
    The derivative at point X[i] is calculated using (Y[i] - Y[i-1]) / (X[i] - X[i-1]).

    Args:
        X: A NumPy array or list of X-coordinates (e.g., Time).
        Y: A NumPy array or list of Y-coordinates (e.g., Response).

    Returns:
        A pandas DataFrame with 'Time', 'Y', and 'Left_Derivative' columns.
        The 'Left_Derivative' for the first point will be NaN.
    """
    left_derivs = []

    # The first point X[0] does not have a "left derivative" from a previous point.
    left_derivs.append(np.nan)

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
        'Left_Derivative': left_derivs
    })

    return df


def remove_outliers_by_region(R: tuple, response_column_name: str = '', Y: pd.DataFrame = pd.DataFrame([])) -> pd.DataFrame:
    """
    Removes outliers from a specified response column within a given time region
    using the k-standard deviation rule (k=3).

    Args:
        R: A tuple (t0, tf) defining the time region.
        response_column_name: The name of the column to detect outliers in.
        Y: The input pandas DataFrame containing 'Time' and the response column.

    Returns:
        A new DataFrame with outliers removed from the specified region,
        or an empty DataFrame if input is invalid or no data in region.
    """
    # --- Input Validation ---
    # Check if the input DataFrame is empty or if the specified response column does not exist.
    if Y.empty or response_column_name not in Y.columns:
        logging.warning(f"DataFrame is empty or column '{response_column_name}' not found. Returning empty DataFrame.")
        return pd.DataFrame([])

    # Ensure the response column is numeric, as outlier detection typically requires numeric data.
    if not pd.api.types.is_numeric_dtype(Y[response_column_name]):
        logging.warning(f"Column '{response_column_name}' is not numeric. Cannot detect outliers. Returning original DataFrame copy.")
        return Y.copy()  # Return a copy as no changes were made

    # --- Time-based Filtering ---
    # Filter the DataFrame to include only rows within the specified time region [R[0], R[1]].
    # Assumes 'Time' is a column in the DataFrame Y.
    time_filtered_Y = get_region(df=Y, t0=R[0], tf=R[1])

    # If, after time filtering, the DataFrame becomes empty, there's nothing to process.
    if time_filtered_Y.empty:
        logging.warning(f"No data found within the specified time region {R}. Returning empty DataFrame.")
        return pd.DataFrame([])

    # --- Outlier Detection Calculation ---
    # Get the data for the response column from the time-filtered DataFrame.
    response_data = time_filtered_Y[response_column_name]

    # Calculate basic statistics for the response data within the region.
    Y_bar = response_data.mean()
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
    Y_prime = time_filtered_Y[is_not_outlier].copy()  # .copy() to avoid SettingWithCopyWarning

    return Y_prime


def sanitize_response_name(response: str) -> str:
    """
    Sanitizes a string to be suitable for use as a column name or identifier.
    Replaces slashes, parentheses, commas, spaces, and dashes with underscores.

    Args:
        response: The input string (e.g., a raw response name).

    Returns:
        A sanitized string.
    """
    # Replace slashes with underscores
    clean = response.replace('/', '_')
    # Remove parentheses and commas
    clean = re.sub(r'[(),]', '', clean)
    # Replace spaces and dashes with underscores
    clean = re.sub(r'[\s\-]+', '_', clean)
    return clean.strip('_') # remove leading/trailing underscores if any
