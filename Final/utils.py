import pandas as pd
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

def load_ec_csv(path_to_csv_file: str, ):
    # Step 1: Read data
    # Identify potential value
    ec_df = pd.read_csv(path_to_csv_file)
    # Normalize time
    ec_df['Time/sec'] = ec_df['Time/sec'] - ec_df['Time/sec'].min()

    # Clean data
    row_index = ec_df[ec_df.apply(lambda row: row.astype(str).str.contains('Time/sec').any(), axis=1)].index[0]

    ec_df_aux_data = ec_df.iloc[:row_index]

    ec_df_cleaned = ec_df.iloc[row_index + 1:]

    ec_df_cleaned.columns = ['Time/sec', 'Current/A']
    ec_df_cleaned.set_index('Time/sec', inplace=True)

    ec_df_cleaned['Current/A'] = ec_df_cleaned['Current/A'].astype(float)

    return ec_df_cleaned

def find_intervals(I0=1800, T=3600, cycles=3, v=[-0.4, 0, 0.4]):
    It = I0

    cycles = 3  # change cycle number
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



def replace_sections(tN: dict, ec_df: pd.DataFrame,  ):
    ec_df['Voltage'] = np.nan

    for t in sorted(tN):  # ensure ascending order
        mask = (ec_df['Time/sec'] <= t) & (ec_df['Voltage'].isna())
        ec_df.loc[mask, 'Voltage'] = tN[t]

    return ec_df
def clean_white_space(df: pd.DataFrame):
    return df.rename(columns=lambda x: x.strip()).sort_values("Time/sec")

def load_trunc_icp_csv(path_to_csv_file: str, del_start: int, response_delay: int) -> tuple[pd.DataFrame, np.ndarray]:
    icp_df = pd.read_csv(path_to_csv_file)

    icp_trunc = icp_df.iloc[del_start:del_start+response_delay]
    icp_trunc['Time'] = icp_trunc['Time'] - (del_start+response_delay)

    return icp_trunc, icp_trunc.to_numpy()


def interpolate(ec_cal: pd.DataFrame, icp_trunc: pd.DataFrame):
    """Interpolates ICP-MS and E-chem data"""
    ec_times = ec_cal['Time/sec'].values
    ec_voltages = ec_cal['Voltage'].values
    ec_currents = ec_cal['Current/A'].values

    fin = []

    for i in range(len(icp_trunc)):
        fin_t = icp_trunc.iloc[i]['Time']

        idx = np.searchsorted(ec_times, fin_t)

        if idx == 0:
            interp_voltage = ec_voltages[0]
            interp_current = ec_currents[0]
        elif idx >= len(ec_times):
            interp_voltage = ec_voltages[-1]
            interp_current = ec_currents[-1]
        else:
            t0, t1 = ec_times[idx - 1], ec_times[idx]
            v0, v1 = ec_voltages[idx - 1], ec_voltages[idx]
            c0, c1 = ec_currents[idx - 1], ec_currents[idx]

            interp_voltage = v0 + (fin_t - t0) * (v1 - v0) / (t1 - t0)
            interp_current = c0 + (fin_t - t0) * (c1 - c0) / (t1 - t0)

        fin.append([fin_t, interp_voltage, interp_current])

    fin_df = pd.DataFrame(fin, columns=['Time', 'Potential/v', 'Current/A_(interp\'d)'])

    # Drop unwanted columns from icp_trunc and reset index
    extra_cols = icp_trunc.drop(columns=['Replicate', 'Reading', 'Time'], errors='ignore').reset_index(drop=True)

    # Concatenate the two DataFrames column-wise
    fin_df = pd.concat([extra_cols, fin_df], axis=1)

    fin_df = fin_df.set_index('Time')
    return fin_df


def plot_amp_voltage(t, fin_df: pd.DataFrame):
    x = np.linspace(0, 2, 100)

    t = fin_df.index
    current = fin_df['Current/A_(interp\'d)']
    potential = fin_df['Potential/v']

    fig, ax1 = plt.subplots(figsize=(10, 7), layout='constrained')

    color = 'tab:red'
    ax1.set_xlabel('t (seconds)')
    ax1.plot(t, current, label='Current', color=color)
    ax1.set_ylabel('Current (amps)')

    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Potential (v)')
    ax2.plot(t, potential, label='Potential', color=color)

    fig.tight_layout()
    ax1.legend()
    ax2.legend()
    return fig, ax1




