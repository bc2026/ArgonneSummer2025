#!/usr/bin/env python3
"""
CSV Time Series Analysis Backend
Flask server to handle CSV data processing and region analysis
"""

PORT = 5001

import glob
import os
import zipfile
import config
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import base64
from datetime import datetime
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from itertools import chain
from utils.FinalUtils import (load_trunc_icp_csv, load_ec_csv, find_intervals, add_potential, CONSTANT_A, clean_white_space, interpolate)

# Set matplotlib backend to non-interactive (prevents GUI issues on macOS)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression

from DetectionTools import *
import sys
from utils.OutlierUtils import left_derivatives, get_region, remove_outliers_by_region, sanitize_response_name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Electron frontend

class CSVAnalyzer:
    def __init__(self):
        self.data = None
        self.time_column = None
        self.response = None
        self.REGIONS_PLOT_PATH = None
        self.FINAL_PLOT_PATH = None
        self.REGRESSION_PLOT_PATH = None
        self.APPLIED_POTENTIAL_PLOT_PATH = None
        self.zipped_files = None
        self.preview_data = None  # Store preview data for later use

    def _update_plot_paths(self):
        """Update plot paths based on current response"""
        if self.response:
            sanitized_response = sanitize_response_name(self.response)
            self.REGIONS_PLOT_PATH = config.FIGURES_PATH.joinpath(f'Regions_Uncleaned_{sanitized_response}.png')
            self.FINAL_PLOT_PATH = config.FIGURES_PATH.joinpath(f'Final_Plot_{sanitized_response}.png')
            self.REGRESSION_PLOT_PATH = config.FIGURES_PATH.joinpath(f'Regression_Plot_{sanitized_response}.png')
            self.APPLIED_POTENTIAL_PLOT_PATH = config.FIGURES_PATH.joinpath(f'Applied_Potential_Plot_{sanitized_response}.png')

    def load_csv_data(self, csv_content, time_column, response):
        """Load and validate CSV data"""
        df = pd.read_csv(io.StringIO(csv_content))
        
        try:
            # Validate columns exist
            if time_column not in df.columns:
                raise ValueError(f"Time column '{time_column}' not found in CSV")
            if response not in df.columns:
                raise ValueError(f"Data column '{response}' not found in CSV")
            
            # Normalize the time column to start from zero
            df[time_column] = df[time_column] - df[time_column].min()
            
            # Rename time column to 'Time' for consistency with rest of codebase
            if time_column != 'Time':
                df = df.rename(columns={time_column: 'Time'})
            
            # Store data
            self.data = df
            self.time_column = 'Time'  # Always use 'Time' internally after renaming
            self.response = response
            self._update_plot_paths()

            
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise

    def plot_regression(self, regional_means: list, custom_indices=None, custom_axis=None):
        """
        Plots regression line (regional mean ~ region)
        """
        # Use custom indices if provided, otherwise use default region numbering
        if custom_indices and len(custom_indices) == len(regional_means):
            x_values = np.array(custom_indices)

            if custom_axis:
                x_label = str(custom_axis)
            else:
                x_label = 'No Label Was Given'
        else:
            x_values = np.arange(len(regional_means))
            x_label = 'Region'

        fig, ax1 = plt.subplots(figsize=(10, 7))

        # Plot the scatter points
        ax1.scatter(x_values, regional_means)
        
        # Calculate regression
        m, b = np.polyfit(x_values, regional_means, 1)

        model = LinearRegression()

        X = x_values.reshape(-1, 1)
        y = np.array(regional_means)

        model.fit(X, y)

        r_squared = model.score(X, y)

        # Find regression equation
        if b >= 0:
            line_label = f'y = {m:.4f}x + {b:.4f}'
        else:
            line_label = f'y = {m:.4f}x - {-b:.4f}'  # Use -b to show a positive number after the minus sign

        # Plot regression line
        ax1.plot(x_values, m * x_values + b, color='red', label=f'{line_label}, r²={r_squared:.4f}')

        plt.legend()

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(f'{self.response} (mean)')

        # Improve tick display
        if custom_indices:
            ax1.set_xticks(x_values)
            ax1.set_xticklabels([str(int(x)) if x == int(x) else str(x) for x in x_values], rotation=90)
        else:
            ax1.set_xticks(range(len(x_values)))
            ax1.set_xticklabels(x_values, rotation=90)
            
        plt.tight_layout()
        plt.savefig(self.REGRESSION_PLOT_PATH, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_column_figure(self, merged_points: list[tuple]):
        """
        Plots a [(num_regions + ncols - 1) // ncols] rows figure of each region
        with original data, removed outliers, means (before and after outlier removal).
        """

        regional_means = []

        num_regions = len(merged_points)
        ncols = 3  # You can change the number of columns
        nrows = (num_regions + ncols - 1) // ncols

        offset = 15  # 15 seconds offset to

        final_df = pd.DataFrame()

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 4 * nrows))
        axes_flat = axes.flatten()  # Flatten the 2D axes array for easier iteration

        for i, (x0, x1) in enumerate(merged_points):
            # Get the current Axes object for this iteration from the flattened array
            if i < len(axes_flat):
                current_ax = axes_flat[i]
            else:
                # This case should ideally not happen if nrows is calculated correctly,
                # but provides a safeguard if more plots are attempted than subplots available.
                logging.warning(f"Not enough subplots for region {i + 1}. Skipping plot.")
                continue

            # Call your outlier removal function for the current region
            if self.data is not None and self.response is not None:
                Y_prime = remove_outliers_by_region(R=(x0 + offset, x1 - offset), response_column_name=self.response, Y=self.data)
            else:
                Y_prime = pd.DataFrame()

            # print(f'New Region {i+1}: ({val['Start'] + offset}, {val['End'] - offset})')

            # Plot the results on the current subplot
            if not Y_prime.empty:
                # Plot the data after outlier removal
                current_ax.plot(Y_prime['Time'], Y_prime[self.response], label='Outliers Removed', color='blue', alpha=0.8)

                # Plot original data for comparison (only within the current region)

                original_region_data = get_region(self.data, x0, x1)

                current_ax.plot(original_region_data['Time'], original_region_data[self.response],
                                '--', color='yellow', alpha=0.7, label='Original Data')

                region_mean = Y_prime[self.response].mean()
                original_mean = original_region_data[self.response].mean()

                current_ax.axhline(y=region_mean, color='#FF69B4', linestyle='-.', label=f'mean={region_mean}')
                current_ax.axhline(y=original_mean, color='red', linestyle='--', label=f'original mean={original_mean}')

                regional_means.append(region_mean)

                current_ax.legend(fontsize='small')
            else:
                current_ax.text(0.5, 0.5, 'No Data or Outliers Removed\nfor this Region',
                                horizontalalignment='center', verticalalignment='center',
                                transform=current_ax.transAxes, fontsize=10, color='red')

            # Set titles and labels for the current subplot
            current_ax.set_title(f'Region {i + 1}: {x0} to {x1}')
            current_ax.set_xlabel('Time (s)')
            current_ax.set_ylabel(f'{self.response}')
            current_ax.grid(True, linestyle='--', alpha=0.6)  # Add a grid for better readability

            final_df = pd.concat([final_df, Y_prime], ignore_index=True)

        # This prevents empty plots from appearing if your `area_df` has fewer regions than the grid size
        for j in range(num_regions, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        plt.savefig(self.REGIONS_PLOT_PATH, bbox_inches='tight', dpi=300)
        plt.close()

        # Store final_df as instance variable and save as CSV
        self.final_df = final_df
        
        # Save final_df as CSV file
        if not final_df.empty:
            csv_filename = f'{sanitize_response_name(self.response)}_outliers_removed_data.csv'
            csv_path = config.FIGURES_PATH / csv_filename
            final_df.to_csv(csv_path, index=False)
            logging.info(f"Saved final_df to {csv_path}")
        else:
            logging.warning("final_df is empty, not saving CSV")

        return regional_means

    def plot_final(self, df_c: pd.DataFrame):
        """
        Plots {response} against time in Orange
        Plots Left-Derivatives against time in Blue
        """

        left_derivatives_df_c = left_derivatives(df_c['Time'].to_numpy(), df_c[self.response].to_numpy())

        fig, ax1 = plt.subplots(figsize=(18, 12), layout='constrained')

        plt.scatter(left_derivatives_df_c['Time'], left_derivatives_df_c['Left_Derivative'], 
                   alpha=0.7, label='Left Derivatives', color='blue')
        plt.scatter(df_c['Time'], df_c[self.response], 
                   alpha=0.7, label=f'{self.response} Data', color='orange')

        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title(f'Preview: {self.response} vs Time with Left Derivatives')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save derivative and data plot
        plt.savefig(self.FINAL_PLOT_PATH, bbox_inches='tight', dpi=300)
        plt.close()

        return left_derivatives_df_c

    def get_detected_intervals_notebook(self, shifts_index_list, first_region):
        """Get detected intervals exactly as in notebook - ensuring continuity"""
        result = [first_region]
        
        # Ensure continuity by making sure regions connect properly
        if len(shifts_index_list) >= 1:
            # Create a region from first_region end to first shift (eliminates gap)
            result.append((first_region[1], shifts_index_list[0]))
            
            # Then create regions between consecutive shifts
            for i in range(0, len(shifts_index_list) - 1):
                start_p = shifts_index_list[i]
                next_p = shifts_index_list[i + 1]
                result.append((start_p, next_p))
        
        return result

    def create_preview(self, first_region_end_index):
        """Create preview of the final plot for user feedback - exact notebook implementation"""
        try:
            if self.data is None or self.time_column is None or self.response is None:
                raise ValueError("Data not loaded properly")

            # Prepare data following notebook approach exactly
            df = self.data
            response = self.response

            # Filter to only include data up to maximum response value (notebook step)
            df_c = df.loc[:, df.columns.intersection(['Time', f'{response}'])]
            df_c_response_idx_max = df_c.idxmax()[response]
            df_c = df_c.iloc[:df_c_response_idx_max + 1]

            # Normalize time data by subtracting minimum (notebook approach)
            df_c = df_c.copy()
            df_c['Time'] = df_c['Time'] - df_c['Time'].min()
            
            # Update first_region_end_index to normalized time
            first_region_end_normalized = first_region_end_index - df['Time'].min()

            # Define first region exactly as in notebook
            first_region = (0.0, first_region_end_normalized)

            # Calculate left derivatives exactly as in notebook
            left_derivatives_df_c = left_derivatives(df_c['Time'].to_numpy(), df_c[response].to_numpy())
            
            # Filter to only consider data after first region end
            left_derivatives_df_c = left_derivatives_df_c[left_derivatives_df_c['Time'] > first_region[1]]
            left_derivatives_df_c.set_index(['Time'], inplace=True)
            
            # Find shifts where left derivative > Y (exact notebook logic)
            mask1 = (left_derivatives_df_c['Left_Derivative'] > left_derivatives_df_c['Y'])
            shifts = left_derivatives_df_c[mask1]
            mask2 = (shifts.index > first_region[1])
            shifts = shifts[mask2]
            
            # Get detected intervals exactly as in notebook
            detected_intervals = self.get_detected_intervals_notebook(shifts.index.tolist(), first_region)
            
            # Add final region from last shift to end of data (exact notebook logic)
            if len(detected_intervals) > 0:
                last_end = detected_intervals[-1][1]
                max_time = df_c['Time'].max()
                print(f"Adding final region from {last_end} to {max_time}")
                detected_intervals.append((last_end, max_time))
            
            print(f"Final detected_intervals: {detected_intervals}")
            
            # Apply default p-value merging to make preview cleaner (like live preview)
            default_p_value = 70  # Same default as live preview
            merged_regions = merge_close_points(df=df_c, regions=detected_intervals, p=default_p_value, response=response)
            
            print(f"Merged regions with p-value {default_p_value}: {merged_regions}")
            
            # Create preview plot with merged regions for cleaner display
            self.plot_final_notebook_style(df, response, first_region_end_index, merged_regions)

            # Store preview data for later use (store both raw and merged)
            self.preview_data = {
                'df': df,
                'df_c': df_c,
                'detected_intervals': detected_intervals,  # Keep raw intervals for live preview
                'merged_regions': merged_regions,  # Store merged regions
                'first_region_end_normalized': first_region_end_normalized,
                'shifts': shifts.index.tolist()
            }

            # Convert plot to base64 for frontend
            preview_image = self.get_plot_as_base64(self.FINAL_PLOT_PATH)
            
            return {
                'preview_image': preview_image,
                'detected_regions_count': len(merged_regions),  # Show merged count
                'shifts_count': len(shifts)
            }

        except Exception as e:
            logger.error(f"Error creating preview: {str(e)}")
            raise

    def create_preview_with_regions(self, region_start_index=None, region_end_index=None, first_region_end_index=None):
        """Create preview with support for multiple region selection types"""
        # If first_region_end_index is provided, use it
        if first_region_end_index is not None:
            return self.create_preview(first_region_end_index)
        
        # If no first_region_end_index, we need a sensible default for the algorithm
        # The algorithm needs the first region to end somewhere meaningful in the data
        if self.data is not None:
            # Use 20% of the time range as a reasonable default for first region end
            time_min = self.data[self.time_column].min()
            time_max = self.data[self.time_column].max()
            time_range = time_max - time_min
            sensible_first_region_end = time_min + (time_range * 0.2)  # 20% into the data
            
            logger.info(f"No first_region_end_index provided, using 20% of time range: {sensible_first_region_end}")
            return self.create_preview(sensible_first_region_end)
        else:
            raise ValueError("No region boundaries provided and no data loaded")

    def plot_final_notebook_style(self, df, response, first_region_end_index, regions):
        """Plot final preview in notebook style showing regions (can be detected intervals or merged regions)"""
        
        fig, ax1 = plt.subplots(figsize=(18, 12), layout='constrained')

        # Plot original data
        ax1.scatter(df['Time'], df[response], 
                   alpha=0.6, label=f'{response} Data', color='lightblue', s=10)

        # Plot regions as filled regions and boundaries
        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, (start, end) in enumerate(regions):
            color = colors[i % len(colors)]
            
            # Convert normalized time back to original scale for display
            original_min_time = df['Time'].min()
            start_original = start + original_min_time
            end_original = end + original_min_time
            
            # Get data for this region
            region_mask = (df['Time'] >= start_original) & (df['Time'] <= end_original)
            region_data = df[region_mask]
            
            # Determine label based on position
            if i == 0:
                region_label = f'First Region (Index: {i+1})'
            elif i == len(regions) - 1:
                region_label = f'Last Region (Index: {i+1})'
            else:
                region_label = f'Region {i+1} (Index: {i+1})'
            
            if not region_data.empty:
                # Highlight this region
                ax1.scatter(region_data['Time'], region_data[response], 
                           color=color, alpha=0.8, s=20, 
                           label=region_label if i < 9 else None)
                
                # Add region index as text annotation
                mid_x = (start_original + end_original) / 2
                mid_y = region_data[response].median() if not region_data.empty else 0
                ax1.annotate(f'Idx: {i+1}', 
                           xy=(mid_x, mid_y), 
                           xytext=(5, 5), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                           fontsize=10, fontweight='bold', color='white')
            
            # Add vertical lines at boundaries
            ax1.axvline(x=start_original, color=color, linestyle='--', alpha=0.7, linewidth=2)
            ax1.axvline(x=end_original, color=color, linestyle='--', alpha=0.7, linewidth=2)

        # Mark the user-selected first region end with special emphasis
        ax1.axvline(x=first_region_end_index, color='black', linestyle='-', linewidth=4, 
                   label='END of First Region (User Selected)')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Value')
        ax1.set_title(f'Preview: Merged Regions (P-value: 70)\n{len(regions)} regions after merging\nDefault Indices: 1,2,3...')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Save the plot
        plt.savefig(self.FINAL_PLOT_PATH, bbox_inches='tight', dpi=300)
        plt.close()

    def create_live_preview(self, first_region_end_index, p_value, regression_indices=None, regression_axis=None):
        """Create live preview showing merged regions with given p-value - exact notebook implementation"""
        try:
            if self.data is None or self.time_column is None or self.response is None:
                raise ValueError("Data not loaded properly")

            # Use existing preview data if available, otherwise regenerate
            if self.preview_data:
                df = self.preview_data['df']
                df_c = self.preview_data['df_c']
                detected_intervals = self.preview_data['detected_intervals']
                shifts = self.preview_data.get('shifts', [])
                response = self.response
            else:
                # Regenerate using exact notebook approach
                df = self.data
                response = self.response

                # Filter to only include data up to maximum response value (notebook step)
                df_c = df.loc[:, df.columns.intersection(['Time', f'{response}'])]
                df_c_response_idx_max = df_c.idxmax()[response]
                df_c = df_c.iloc[:df_c_response_idx_max + 1]

                # Normalize time data by subtracting minimum (notebook approach)
                df_c = df_c.copy()
                df_c['Time'] = df_c['Time'] - df_c['Time'].min()
                
                # Update first_region_end_index to normalized time
                first_region_end_normalized = first_region_end_index - df['Time'].min()

                # Define first region exactly as in notebook
                first_region = (0.0, first_region_end_normalized)

                # Calculate left derivatives exactly as in notebook
                left_derivatives_df_c = left_derivatives(df_c['Time'].to_numpy(), df_c[response].to_numpy())
                
                # Filter to only consider data after first region end
                left_derivatives_df_c = left_derivatives_df_c[left_derivatives_df_c['Time'] > first_region[1]]
                left_derivatives_df_c.set_index(['Time'], inplace=True)
                
                # Find shifts where left derivative > Y (exact notebook logic)
                mask1 = (left_derivatives_df_c['Left_Derivative'] > left_derivatives_df_c['Y'])
                shifts = left_derivatives_df_c[mask1]
                mask2 = (shifts.index > first_region[1])
                shifts = shifts[mask2]
                
                # Get detected intervals exactly as in notebook
                detected_intervals = self.get_detected_intervals_notebook(shifts.index.tolist(), first_region)
                
                # Add final region from last shift to end of data (exact notebook logic)
                if len(detected_intervals) > 0:
                    last_end = detected_intervals[-1][1]
                    max_time = df_c['Time'].max()
                    print(f"Live preview - Adding final region from {last_end} to {max_time}")
                    detected_intervals.append((last_end, max_time))
                
                print(f"Live preview - Final detected_intervals: {detected_intervals}")

            # Apply p-value to merge regions exactly as in notebook
            merged_regions = merge_close_points(df=df_c, regions=detected_intervals, p=p_value, response=response)
            
            # Extract merged points list as in notebook
            merged_points_list = sorted(list(set(tuple(chain.from_iterable(merged_regions)))))

            # Create a visualization showing the merged regions with notebook style
            self.plot_live_preview_notebook_style(df, merged_regions, p_value, response, first_region_end_index, regression_indices)

            # Convert plot to base64 for frontend
            preview_image = self.get_plot_as_base64(self.FINAL_PLOT_PATH)
            
            return {
                'preview_image': preview_image,
                'merged_regions_count': len(merged_regions),
                'shifts_count': len(shifts) if isinstance(shifts, list) else len(shifts.index) if hasattr(shifts, 'index') else len(shifts)
            }

        except Exception as e:
            logger.error(f"Error creating live preview: {str(e)}")
            raise

    def plot_live_preview_notebook_style(self, df, merged_regions, p_value, response, first_region_end_index, regression_indices=None):
        """Plot live preview showing merged regions in notebook style"""
        
        fig, ax1 = plt.subplots(figsize=(18, 12), layout='constrained')

        # Plot original data
        ax1.scatter(df['Time'], df[response], 
                   alpha=0.6, label=f'{response} Data', color='lightblue', s=10)

        # Plot merged regions with boundaries and highlights
        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, (start, end) in enumerate(merged_regions):
            color = colors[i % len(colors)]
            
            # Convert normalized time back to original scale for display
            original_min_time = df['Time'].min()
            start_original = start + original_min_time
            end_original = end + original_min_time
            
            # Get data for this merged region
            region_mask = (df['Time'] >= start_original) & (df['Time'] <= end_original)
            region_data = df[region_mask]
            
            # Determine label based on position, include regression index if available
            region_index = regression_indices[i] if regression_indices and i < len(regression_indices) else (i + 1)
            if i == 0:
                region_label = f'First Merged Region (Index: {region_index})'
            elif i == len(merged_regions) - 1:
                region_label = f'Last Merged Region (Index: {region_index})'
            else:
                region_label = f'Merged Region {i+1} (Index: {region_index})'
            
            if not region_data.empty:
                # Highlight this merged region
                ax1.scatter(region_data['Time'], region_data[response], 
                           color=color, alpha=0.8, s=20, 
                           label=region_label if i < 9 else None)
                
                # Add vertical lines at boundaries
                ax1.axvline(x=start_original, color=color, linestyle='--', alpha=0.7, linewidth=2)
                ax1.axvline(x=end_original, color=color, linestyle='--', alpha=0.7, linewidth=2)
                
                # Add region index as text annotation
                mid_x = (start_original + end_original) / 2
                mid_y = region_data[response].median() if not region_data.empty else 0
                ax1.annotate(f'Idx: {region_index}', 
                           xy=(mid_x, mid_y), 
                           xytext=(5, 5), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                           fontsize=10, fontweight='bold', color='white')

        # Mark the user-selected first region end with special emphasis
        ax1.axvline(x=first_region_end_index, color='black', linestyle='-', linewidth=4, 
                   label='END of First Region (User Selected)')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Value')
        
        # Include regression indices info in title
        indices_text = f"Custom Indices: {regression_indices}" if regression_indices else "Default Indices: 1,2,3..."
        ax1.set_title(f'Live Preview: Merged Regions (P-value: {p_value})\n{len(merged_regions)} regions after merging\n{indices_text}')
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Save the plot
        plt.savefig(self.FINAL_PLOT_PATH, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_applied_potential(self, df, response, first_region_end_index, del_start=5, response_delay=300/60, FLOW_RATE=1.0, amount=1.0):
        """Plot applied potential and response against time"""
        path_to_ec_csv = './data/Example/applied_potential.txt'
        
        # Load ICP-MS data properly
        df = load_trunc_icp_csv(self.data, del_start=del_start, response_delay=response_delay, t=self.time_column)
        
        df[f'{response}-ppb'] = df[response] / CONSTANT_A
        df[f'{response}-ug/hr'] = df[f'{response}-ppb'] * 0.01 * 60 * FLOW_RATE
        df.set_index(self.time_column, inplace=True)

        ec_df = load_ec_csv(path_to_ec_csv, t=self.time_column)
        tN = find_intervals(I0=ec_df.index.min(), T=1, cycles=3, v=[0.4, 0.95])
        ec_df = add_potential(tN, ec_df=ec_df)
        
        ec_df_aligned = ec_df.reindex(df.index, method='pad')

    
        # Extract matching values
        t = df.index
        y1 = df[f'{response}-ug/hr'].values
        y2 = ec_df_aligned['Potential (v)'].values

        fig, ax1 = plt.subplots(figsize=(16, 11), layout='constrained')

        # Left Y-axis: Pt
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(f'{response} (µg/hr)', color='tab:red')
        line1, = ax1.plot(t, y1, color='tab:red', label=f'{response} (µg/hr)')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        # Right Y-axis: Potential
        ax2 = ax1.twinx()
        ax2.set_ylabel('Potential (V)', color='tab:blue')
        line2, = ax2.step(t, y2, color='tab:blue', label='Applied Potential', where='post')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Legend
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='lower right')

        plt.xlim(5, 305/60)  # Only show the range t=2 to t=5
        plt.title(f'{response} vs Applied Potential Over Time')
        plt.savefig(self.APPLIED_POTENTIAL_PLOT_PATH, bbox_inches='tight', dpi=300)
        plt.close()



    def plot_merged_regions_preview(self, df, merged_regions, p_value, response):
        """Plot preview showing merged regions with visual boundaries"""
        
        fig, ax1 = plt.subplots(figsize=(18, 12), layout='constrained')

        # Plot original data
        ax1.scatter(df['Time'], df[response], 
                   alpha=0.6, label=f'{response} Data', color='lightblue', s=10)

        # Plot region boundaries and highlight merged regions
        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, (start, end) in enumerate(merged_regions):
            color = colors[i % len(colors)]
            
            # Get data for this region
            region_mask = (df['Time'] >= start) & (df['Time'] <= end)
            region_data = df[region_mask]
            
            # Determine label based on position
            if i == 0:
                region_label = 'First Merged Region'
            elif i == len(merged_regions) - 1:
                region_label = 'Last Merged Region'
            else:
                region_label = f'Merged Region {i+1}'
            
            if not region_data.empty:
                # Highlight this merged region
                ax1.scatter(region_data['Time'], region_data[response], 
                           color=color, alpha=0.8, s=20, 
                           label=region_label)
                
                # Add vertical lines at boundaries
                ax1.axvline(x=start, color=color, linestyle='--', alpha=0.7, linewidth=2)
                ax1.axvline(x=end, color=color, linestyle='--', alpha=0.7, linewidth=2)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Value')
        ax1.set_title(f'Live Preview: Merged Regions (P-value: {p_value})\n{len(merged_regions)} regions after merging')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Save the plot
        plt.savefig(self.FINAL_PLOT_PATH, bbox_inches='tight', dpi=300)
        plt.close()

    def get_analysis_images(self):
        """Get all analysis images as base64 encoded strings"""
        images = {}
        
        # Get regions plot
        if self.REGIONS_PLOT_PATH and self.REGIONS_PLOT_PATH.exists():
            images['regions_plot'] = self.get_plot_as_base64(self.REGIONS_PLOT_PATH)
            
        # Get final plot
        if self.FINAL_PLOT_PATH and self.FINAL_PLOT_PATH.exists():
            images['final_plot'] = self.get_plot_as_base64(self.FINAL_PLOT_PATH)
            
        # Get regression plot
        if self.REGRESSION_PLOT_PATH and self.REGRESSION_PLOT_PATH.exists():
            images['regression_plot'] = self.get_plot_as_base64(self.REGRESSION_PLOT_PATH)
        
        # Get applied potential plot
        if self.APPLIED_POTENTIAL_PLOT_PATH and self.APPLIED_POTENTIAL_PLOT_PATH.exists():
            images['applied_potential_plot'] = self.get_plot_as_base64(self.APPLIED_POTENTIAL_PLOT_PATH)

        return images

    def analyze_regions(self, first_region_end_index, p_value=70, regression_indices=None, regression_axis=None):
        """Perform statistical analysis on each region with configurable p-value"""
        try:
            if self.data is None or self.time_column is None or self.response is None:
                raise ValueError("Data incomplete. Failed analysis.")

            # Use preview data if available, otherwise regenerate
            if self.preview_data:
                df = self.preview_data['df']
                detected_intervals = self.preview_data['detected_intervals']
                response = self.response
            else:
                # Regenerate if no preview data
                df = self.data
                response = self.response

                df_c = df.loc[:, df.columns.intersection(['Time', f'{response}'])]
                df_c_response_idx_max = df_c.idxmax()[response]
                df_c = df_c.loc[:df_c_response_idx_max]
                df_c = df_c[df_c['Time'] > first_region_end_index].reset_index(drop=True)

                first_region = [(0, first_region_end_index)]

                left_derivatives_df_c = self.plot_final(df_c)
                shifts = left_derivatives_df_c[left_derivatives_df_c['Left_Derivative'] > left_derivatives_df_c['Y']]
                
                # Extract just the Time values from the shifts DataFrame
                shifts = shifts.reset_index(drop=True)
                shifts = shifts['Time']

                detected_intervals = get_detected_intervals(shifts, first_region)

            # Use the provided p_value for merging
            merged_points = merge_close_points(df=df, regions=detected_intervals, p=p_value, response=response)

            regional_means = self.plot_column_figure(merged_points=merged_points)

            self.plot_regression(regional_means=regional_means, custom_indices=regression_indices, custom_axis=regression_axis)

            self.plot_applied_potential(df=df, response=response, first_region_end_index=first_region_end_index)

            logging.info("Plotting complete...")

            # Get analysis images as base64
            analysis_images = self.get_analysis_images()

            logger.info(f"Completed analysis for {len(detected_intervals)} detected intervals, {len(merged_points)} merged regions")

            logger.info(f"Zipping files....")

            zip_path = self.zip_files()

            if not zip_path:
                logging.error("Failed to zip all files.")
                raise ValueError("Failed to zip all files.")

            return {
                'response': self.response,
                'regions': merged_points,
                'regional_means': regional_means,
                'detected_regions_count': len(detected_intervals),
                'merged_regions_count': len(merged_points),
                'zip_path': str(zip_path),
                'analysis_images': analysis_images
            }
        except Exception as e:
            logger.error(f"Error analyzing regions: {str(e)}")
            raise
    
    def _format_value(self, value):
        """Format a value for JSON serialization"""
        if pd.isna(value):
            return None
        elif isinstance(value, (np.integer, int)):
            return int(value)
        elif isinstance(value, (np.floating, float)):
            return float(value)
        elif isinstance(value, pd.Timestamp):
            return value.isoformat()
        else:
            return str(value)

    def get_plot_as_base64(self, plot_path):
        """Convert plot to base64 string for frontend display"""
        try:
            with open(plot_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                return img_base64
        except Exception as e:
            logger.error(f"Error converting plot to base64: {str(e)}")
            return None

    def zip_files(self):
        """Zip all generated files including plots and CSV data"""
        try:
            zip_filename = f'{sanitize_response_name(self.response)}_export_{datetime.now().strftime("%Y%m%d-%H%M%S")}.zip'
            zip_path = config.FIGURES_PATH / zip_filename

            # Ensure the source directory exists
            if not os.path.isdir(config.FIGURES_PATH):
                print(f"Error: Source directory '{config.FIGURES_PATH}' does not exist.")
                return None
            
            if not os.path.isdir(config.EXPORT_DATA_PATH):
                logging.error(f"Error: Source directory '{config.EXPORT_DATA_PATH}' does not exist.")
                return None

            # Create a ZipFile object in write mode ('w')
            # 'zipfile.ZIP_DEFLATED' specifies compression (recommended)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Find all PNG files in the figures directory
                png_files = list(config.FIGURES_PATH.glob('*.png'))
                for file_path in png_files:
                    # Check if the file exists before adding
                    if file_path.exists():
                        zipf.write(file_path, arcname=file_path.name)
                        logging.info(f"Added '{file_path.name}' to '{zip_filename}'")
                    else:
                        logging.warning(f"Warning: File '{file_path}' not found, skipping.")

                # Find all CSV files in the figures directory
                csv_files = list(config.FIGURES_PATH.glob('*.csv'))
                for file_path in csv_files:
                    # Check if the file exists before adding
                    if file_path.exists():
                        zipf.write(file_path, arcname=file_path.name)
                        logging.info(f"Added '{file_path.name}' to '{zip_filename}'")
                    else:
                        logging.warning(f"Warning: CSV file '{file_path}' not found, skipping.")

            self.zipped_files = zip_path
            return zip_path

        except Exception as e: 
            logging.error(f"An error occurred: {e}")
            return None



    def clean_shift_points(self, shifts_index_list, max_time):
        """
        Cleans up shift points by removing consecutive detections and
        ensuring they are not too close to the start or end of the data.
        """
        cleaned_shifts = []
        prev_shift = -float('inf') # Initialize with a value before the first shift
        
        for shift_point in shifts_index_list:
            # Skip if the shift is too close to the start or end of the data
            if shift_point < 0.0 or shift_point > max_time:
                continue

            # Skip if the shift is too close to the previous shift
            if shift_point - prev_shift < (max_time - 0.0) * 0.01:  # 1% of total time range (more aggressive)
                continue

            cleaned_shifts.append(shift_point)
            prev_shift = shift_point
        
        print(f"Cleaned up {len(shifts_index_list)} shifts to {len(cleaned_shifts)} unique shifts.")
        return cleaned_shifts

    def get_default_region_selections(self):
        """Calculate default region selections based on the data"""
        if self.data is None or self.time_column is None or self.response is None:
            return None
        
        # Start region: minimum time
        start_time = self.data[self.time_column].min()
        
        # End region: maximum time
        end_time = self.data[self.time_column].max()
        
        return {
            'region_start_index': float(start_time),
            'region_end_index': float(end_time)
            # first_region_end_index removed - let user choose manually
        }

# Global analyzer instance
analyzer = CSVAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'CSV Time Series Analysis Backend'
    })

@app.route('/preview', methods=['POST'])
def preview_analysis():
    """Preview endpoint to show final plot before region analysis"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['csv_content', 'time_column', 'response']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        csv_content = data['csv_content']
        time_column = data['time_column']
        response = data['response']
        
        # Get region selection parameters (all optional)
        region_start_index = data.get('region_start_index')
        region_end_index = data.get('region_end_index')
        first_region_end_index = data.get('first_region_end_index')
        
        # Convert to float if provided
        if region_start_index is not None:
            region_start_index = float(region_start_index)
        if region_end_index is not None:
            region_end_index = float(region_end_index)
        if first_region_end_index is not None:
            first_region_end_index = float(first_region_end_index)
        
        logger.info(f"Received preview request - Region start: {region_start_index}, Region end: {region_end_index}, First region end: {first_region_end_index}")
        
        # Load CSV data
        analyzer.load_csv_data(csv_content, time_column, response)
        
        # Create preview (use first_region_end_index as fallback for compatibility)
        preview_result = analyzer.create_preview_with_regions(
            region_start_index=region_start_index,
            region_end_index=region_end_index,
            first_region_end_index=first_region_end_index
        )
        
        response_data = {
            'success': True,
            'preview_image': preview_result['preview_image'],
            'detected_regions_count': preview_result['detected_regions_count'],
            'shifts_count': preview_result['shifts_count'],
            'metadata': {
                'total_data_points': len(analyzer.data) if analyzer.data is not None else 0,
                'time_column': time_column,
                'response': response,
                'first_region_end_index': first_region_end_index,
                'processing_timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info("Preview created successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error creating preview: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to create preview'}), 500

@app.route('/live-preview', methods=['POST'])
def live_preview_analysis():
    """Live preview endpoint to show updated region merging with different p-values"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['csv_content', 'time_column', 'response', 'p_value']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        csv_content = data['csv_content']
        time_column = data['time_column']
        response = data['response']
        p_value = float(data['p_value'])
        
        # Get region selection parameters (all optional)
        region_start_index = data.get('region_start_index')
        region_end_index = data.get('region_end_index')
        first_region_end_index = data.get('first_region_end_index')
        
        # Get regression indices (optional)
        regression_indices = data.get('regression_indices')
        regression_axis = data.get('regression_axis')

        # Get E-chem data
        flow_rate = int(data.get('flow_rate'))
        del_start = int(data.get('del_start'))
        response_delay = int(data.get('response_delay'))
        
        logging.info(f'flow_rate: {flow_rate}\n del_start: {del_start}\n response_delay: {response_delay}')



        # Convert to float if provided
        if region_start_index is not None:
            region_start_index = float(region_start_index)
        if region_end_index is not None:
            region_end_index = float(region_end_index)
        if first_region_end_index is not None:
            first_region_end_index = float(first_region_end_index)
        
        logger.info(f"Received live preview request - Region start: {region_start_index}, Region end: {region_end_index}, First region end: {first_region_end_index}, P-value: {p_value}, Regression indices: {regression_indices}")
        
        # Load CSV data if not already loaded
        if analyzer.data is None:
            analyzer.load_csv_data(csv_content, time_column, response)
        
        # Verify data was loaded successfully
        if analyzer.data is None:
            return jsonify({'error': 'Failed to load CSV data'}), 500
        
        # Create live preview with p-value applied
        if first_region_end_index is not None:
            boundary_index = first_region_end_index
        else:
            # Use 20% of the time range as a reasonable default for first region end
            time_min = analyzer.data[time_column].min()
            time_max = analyzer.data[time_column].max()
            time_range = time_max - time_min
            boundary_index = time_min + (time_range * 0.2)  # 20% into the data
            logger.info(f"No first_region_end_index provided for live preview, using 20% of time range: {boundary_index}")
        
        preview_result = analyzer.create_live_preview(boundary_index, p_value, regression_indices, regression_axis)
        
        response_data = {
            'success': True,
            'preview_image': preview_result['preview_image'],
            'detected_regions_count': preview_result['merged_regions_count'],
            'shifts_count': preview_result['shifts_count'],
            'p_value': p_value,
            'metadata': {
                'processing_timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Live preview created successfully with p-value {p_value}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error creating live preview: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to create live preview'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_csv():
    """Main endpoint to receive CSV data and perform region analysis"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['csv_content', 'time_column', 'response']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        csv_content = data['csv_content']
        time_column = data['time_column']
        response = data['response']
        p_value = float(data.get('p_value', 70))  # Default to 70 if not provided
        
        # Get region selection parameters (all optional)
        region_start_index = data.get('region_start_index')
        region_end_index = data.get('region_end_index')
        first_region_end_index = data.get('first_region_end_index')
        
        # Get regression indices (optional)
        regression_indices = data.get('regression_indices')
        regression_axis = data.get('regression_axis')
        
        # Convert to float if provided
        if region_start_index is not None:
            region_start_index = float(region_start_index)
        if region_end_index is not None:
            region_end_index = float(region_end_index)
        if first_region_end_index is not None:
            first_region_end_index = float(first_region_end_index)
        
        logger.info(f"Received analysis request - Region start: {region_start_index}, Region end: {region_end_index}, First region end: {first_region_end_index}, P-value: {p_value}, Regression indices: {regression_indices}")
        
        # Load CSV data (if not already loaded from preview)
        if analyzer.data is None:
            analyzer.load_csv_data(csv_content, time_column, response)

        # Verify data was loaded successfully
        if analyzer.data is None:
            return jsonify({'error': 'Failed to load CSV data'}), 500

        # Determine the primary boundary to use for analysis
        if first_region_end_index is not None:
            boundary_index = first_region_end_index
        else:
            # Use 20% of the time range as a reasonable default for first region end
            time_min = analyzer.data[time_column].min()
            time_max = analyzer.data[time_column].max()
            time_range = time_max - time_min
            boundary_index = time_min + (time_range * 0.2)  # 20% into the data
            logger.info(f"No first_region_end_index provided for analysis, using 20% of time range: {boundary_index}")

        # Perform analysis
        analysis_results = analyzer.analyze_regions(
            first_region_end_index=boundary_index,
            p_value=p_value,
            regression_indices=regression_indices,
            regression_axis=regression_axis
        )
        
        # Prepare response
        response_data = {
            'success': True,
            'metadata': {
                'total_data_points': len(analyzer.data) if analyzer.data is not None else 0,
                'time_column': time_column,
                'response': response,
                'first_region_end_index': first_region_end_index,
                'p_value': p_value,
                'processing_timestamp': datetime.now().isoformat()
            },
            'analysis': analysis_results
        }
        
        logger.info("Analysis completed successfully")
        return jsonify(response_data)
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error occurred'}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download endpoint to serve zip files"""
    try:
        # Security check: only allow downloading files from the figures directory
        file_path = config.FIGURES_PATH / filename
        
        # Check if file exists and is a zip file
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        if not filename.endswith('.zip'):
            return jsonify({'error': 'Only zip files can be downloaded'}), 400
        
        # Serve the file
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': 'Failed to download file'}), 500

@app.route('/download/image/<image_type>', methods=['GET'])
def download_image(image_type):
    """Download individual analysis images"""
    try:
        if analyzer.data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        # Map image types to file paths
        image_paths = {
            'regions': analyzer.REGIONS_PLOT_PATH,
            'final': analyzer.FINAL_PLOT_PATH,
            'regression': analyzer.REGRESSION_PLOT_PATH
        }
        
        if image_type not in image_paths:
            return jsonify({'error': f'Invalid image type: {image_type}'}), 400
        
        image_path = image_paths[image_type]
        
        if not image_path or not image_path.exists():
            return jsonify({'error': f'Image not found: {image_type}'}), 404
        
        # Generate filename with response name
        response_name = sanitize_response_name(analyzer.response)
        filename_map = {
            'regions': f'{response_name}_regions_plot.png',
            'final': f'{response_name}_final_plot.png',
            'regression': f'{response_name}_regression_plot.png'
        }
        
        return send_file(
            image_path,
            as_attachment=True,
            download_name=filename_map[image_type],
            mimetype='image/png'
        )
    
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        return jsonify({'error': 'Failed to download image'}), 500

@app.route('/defaults', methods=['POST'])
def get_default_selections():
    """Get default region selections for the loaded data"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['csv_content', 'time_column', 'response']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        csv_content = data['csv_content']
        time_column = data['time_column']
        response = data['response']
        
        # Load CSV data
        analyzer.load_csv_data(csv_content, time_column, response)
        
        # Get default selections
        defaults = analyzer.get_default_region_selections()
        
        if defaults is None:
            return jsonify({'error': 'Unable to calculate defaults'}), 500
        
        response_data = {
            'success': True,
            'defaults': defaults,
            'metadata': {
                'total_data_points': len(analyzer.data) if analyzer.data is not None else 0,
                'time_column': time_column,
                'response': response,
                'min_time': float(analyzer.data[time_column].min()),
                'max_time': float(analyzer.data[time_column].max())
            }
        }
        
        logger.info("Default selections calculated successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting default selections: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to get default selections'}), 500

if __name__ == '__main__':
    
    logger.info("Starting CSV Time Series Analysis Backend")
    app.run(debug=True, host='127.0.0.1', port=PORT)