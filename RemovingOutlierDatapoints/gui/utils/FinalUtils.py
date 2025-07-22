import pandas as pd
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

CONSTANT_A = 103054



def load_ec_csv(path_to_csv_file: str, t: str):
    ec_df = pd.read_csv(path_to_csv_file, header=None, delimiter="\t")

    row_index = ec_df[ec_df.apply(lambda row: row.astype(str).str.contains('End Comments').any(), axis=1)].index[0]
    ec_df_cleaned = ec_df.iloc[row_index + 1:]

    split_df = ec_df_cleaned[0].str.split('\t', expand=True)

    ec_df_cleaned = split_df.astype(np.float64)
    ec_df_cleaned.columns = ["E (v)", "I (A/cm2)", t]

    ec_df_cleaned[t] = ec_df_cleaned[t] / 60 # seconds to minutes
    ec_df_cleaned.set_index(t, inplace=True)

    return ec_df_cleaned

def find_intervals(I0=1800, T=3600, cycles=3, v=[-0.4, 0, 0.4]):
    It = I0

    V = v * cycles  # change array for desired set of voltages

    # Start a queue
    Vq = deque()

    n = 0
    N = len(V)

    tN = {}  # time associated mapped to voltage

    while n < N:
        if not tN:
            tN[I0] = V[0]
            continue
        if not Vq:
            Vq = deque(V)

        tN[It] = Vq.popleft()
        It += T
        n += 1

    return tN

def add_potential(tN: dict, ec_df: pd.DataFrame):
    ec_df['Potential (v)'] = np.nan
    thresholds = sorted(tN)

    for i, ti in enumerate(thresholds):
        if i < len(thresholds) - 1:
            ti_next = thresholds[i + 1]
            mask = (ec_df.index >= ti) & (ec_df.index < ti_next)
        else:
            mask = ec_df.index >= ti  # Last tier gets everything else

        ec_df.loc[mask, 'Potential (v)'] = tN[ti]

    return ec_df

def clean_white_space(df: pd.DataFrame, t: str):
    return df.rename(columns=lambda x: x.strip()).sort_values(t)

def load_trunc_icp_csv(icp_df: pd.DataFrame, del_start: int, response_delay: int, t: str) -> tuple[pd.DataFrame, np.ndarray]:
    del_start = int(del_start)
    response_delay = int(response_delay)


    icp_df[t] = (icp_df['Time'] - icp_df['Time'].min()) / 60 # seconds to minutes

    icp_df = clean_white_space(icp_df, t=t)
    icp_df[t] = icp_df[t] - icp_df[t].min()

    icp_trunc = icp_df.iloc[del_start:del_start+response_delay]

    # Convert to minutes starting from 0 (align with EC data)
    icp_trunc = icp_trunc.copy()  # Avoid SettingWithCopyWarning

    return icp_trunc


def interpolate(ec_df: pd.DataFrame, icp_trunc_df: pd.DataFrame):
    """Interpolates ICP-MS and E-chem data"""
    ec_times = ec_df.index.to_numpy(dtype=float)  # Assumes ec_df index is float (e.g., seconds)
    ec_potentials = ec_df['E (v)'].values
    ec_density = ec_df['I (A/cm2)'].values
    fin = []

    for i in range(len(icp_trunc_df)):
        fin_t = icp_trunc_df.index[i]

        idx = np.searchsorted(ec_times, fin_t)

        if idx == 0:
            interp_potential = ec_potentials[0]
            interp_density = ec_density[0]
        elif idx >= len(ec_times):
            interp_potential = ec_potentials[-1]
            interp_density = ec_density[-1]
        else:
            t0, t1 = ec_times[idx - 1], ec_times[idx]
            v0, v1 = ec_potentials[idx - 1], ec_potentials[idx]
            d0, d1 = ec_density[idx - 1], ec_density[idx]

            interp_density = d0 + (fin_t - t0) * (d1 - d0) / (t1 - t0)
            interp_potential = v0 + (fin_t - t0) * (v1 - v0) / (t1 - t0)

        fin.append([fin_t, interp_potential, interp_density])

    if not fin:
        raise ValueError('No data found')

    fin_df = pd.DataFrame(fin, columns=['Time', 'E (v)', 'I (A/cm2)'])

    # Drop unwanted columns from icp_trunc and reset index
    extra_cols = icp_trunc_df.drop(columns=['Replicate', 'Reading'], errors='ignore').reset_index(drop=True)

    # Concatenate the two DataFrames column-wise
    fin_df = pd.concat([extra_cols, fin_df], axis=1)

    # Set as index
    fin_df = fin_df.set_index('Time')

    return fin_df

