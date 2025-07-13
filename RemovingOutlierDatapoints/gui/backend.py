#!/usr/bin/env python3
"""
CSV Time Series Analysis Backend
Flask server to handle CSV data processing and region analysis
"""

import glob
import os
import zipfile

from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from datetime import datetime
import traceback
from pathlib import Path




from DetectionTools import *

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
        self.REGIONS_PLOT_PATH = config.FIGURES_PATH.joinpath(f'Regions_Uncleaned_{self.response}.png')
        self.FINAL_PLOT_PATH = config.FIGURES_PATH.joinpath(f'Final_Plot_{self.response}.png')
        self.REGRESSION_PLOT_PATH = config.FIGURES_PATH.joinpath(f'Regression_Plot_{self.response}.png')
        self.zipped_files = None

    def load_csv_data(self, csv_content, time_column, response):
        """Load and validate CSV data"""
        try:
            # Parse CSV content
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Validate columns exist
            if time_column not in df.columns:
                raise ValueError(f"Time column '{time_column}' not found in CSV")
            if response not in df.columns:
                raise ValueError(f"Data column '{response}' not found in CSV")
            
            # Store data
            self.data = df
            self.time_column = time_column
            self.response = response

            
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise

    def plot_regression(self, regional_means: list):
        """
        Plots regression line (regional mean ~ region)
        """
        # Plot regional means
        labels = np.arange(len(regional_means))

        fig, ax1 = plt.subplots(figsize=(10, 7))

        # Plot the bars
        ax1.scatter(labels, regional_means)  # Convert index to string for proper label display
        m, b = np.polyfit(labels, regional_means, 1)

        model = LinearRegression()

        X = labels.reshape(-1, 1)
        y = np.array(regional_means)

        model.fit(X, y)

        r_squared = model.score(X, y)

        # Find regression equation
        if b >= 0:
            line_label = f'y = {m}x + {b}'
        else:
            line_label = f'y = {m}x - {-b}'  # Use -b to show a positive number after the minus sign

        ax1.plot(labels, m * labels + b, color='red', label=f'{line_label}, r²={r_squared}')

        plt.legend()

        ax1.set_xlabel('Region')
        ax1.set_ylabel(f'{self.response} (mean)')

        # Improve tick display
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=90)  # Rotate for better readability
        plt.tight_layout()
        plt.savefig(self.REGRESSION_PLOT_PATH, bbox_inches='tight', dpi=300)

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
            Y_prime = remove_outliers_by_region(R=(x0 + offset, x1 - offset), response_column_name=self.response, Y=self.data)

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

        return regional_means

    def plot_final(self, df_c: pd.DataFrame):
        """
        Plots {response} against time in Orange
        Plots Left-Derivatives against time in Blue
        """

        left_derivatives_df_c = left_derivatives(df_c['Time'].to_numpy(), df_c[self.response].to_numpy())

        fig, ax1 = plt.subplots(figsize=(18, 12), layout='constrained')

        plt.scatter(left_derivatives_df_c['Time'], left_derivatives_df_c['Left_Derivative'])
        plt.scatter(df_c['Time'], df_c[self.response])

        # Save derivative and data plot
        plt.savefig(self.FINAL_PLOT_PATH, bbox_inches='tight', dpi=300)

        return left_derivatives_df_c

    def zip_files(self, source_directory=None):

        try:
            zip_filename = f'{sanitize_response_name(self.response)}_export_{datetime.now().strftime("%Y%m%d-%H%M%S")}.zip'

            # Ensure the source directory exists
            if not os.path.isdir(config.FIGURES_PATH):
                print(f"Error: Source directory '{config.FIGURES_PATH}' does not exist.")
                return None
            
            if not os.path.isdir(config.EXPORT_DATA_PATH):
                logging.error(f"Error: Source directory '{config.EXPORT_DATA_PATH}' does not exist.")
                return None

            # Create a ZipFile object in write mode ('w')
            # 'zipfile.ZIP_DEFLATED' specifies compression (recommended)
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in glob.glob(config.FIGURES_PATH.joinpath('*.png')):
                    # Construct the full path to the file
                    full_file_path = Path(source_directory) /  file_path

                    # Check if the file exists before adding
                    if full_file_path.exists():
                        zipf.write(full_file_path, arcname=file_path)
                        logging.info(f"Added '{file_path}' to '{zip_filename}'")
                    else:
                        logging.warning(f"Warning: File '{full_file_path}' not found, skipping.")

            self.zipped_files = Path(source_directory).joinpath(zip_filename)

        except Exception as e:
            logging.error(f"An error occurred: {e}")


    def analyze_regions(self,first_region_end_index, num_regions) -> list:
        """Perform statistical analysis on each region"""
        try:
            if self.data is None or self.time_column is None or self.response is None or not num_regions:
                raise ValueError("Data incomplete. Failed analysis.")

            # Make sure we only look at after the selected time
            df = self.data
            t = self.time_column
            response = self.response

            df_c = df.loc[:, df.columns.intersection(['Time', f'{response}'])]
            df_c_response_idx_max = df_c.idxmax()[response]
            df_c = df_c.iloc[:df_c_response_idx_max + 1]
            df_c = df_c[df_c['Time'] > first_region_end_index]

            first_region = [(0, first_region_end_index)]

            left_derivatives_df_c = self.plot_final(df_c)
            shifts = left_derivatives_df_c[left_derivatives_df_c['Left_Derivative'] > left_derivatives_df_c['Y']]

            detected_intervals = get_detected_intervals(shifts, first_region)

            merged_points = merge_close_points(df=df, regions=detected_intervals, p=70, response=response)


            regional_means = self.plot_column_figure(merged_points=merged_points)

            self.plot_regression(regional_means=regional_means)

            logging.info("Plotting complete...")

            logger.info(f"Completed analysis for {len(detected_intervals)} regions")

            logger.info(f"Zipping files....")

            self.zip_files(source_directory=config.FIGURES_PATH)

            if not self.zipped_files:
                logging.error("Failed to zip all files.")
                raise ValueError("Failed to zip all files.")

            return self.response
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

@app.route('/analyze', methods=['POST'])
def analyze_csv():
    """Main endpoint to receive CSV data and perform region analysis"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['csv_content', 'time_column', 'response', 'first_region_end_index', 'num_regions']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        csv_content = data['csv_content']

        time_column = data['time_column']
        response = data['response']

        first_region_end_index = int(data['first_region_end_index'])
        num_regions = int(data['num_regions'])
        
        logger.info(f"Received analysis request - Regions: {num_regions}, First region end: {first_region_end_index}")
        
        # Load CSV data
        analyzer.load_csv_data(csv_content, time_column, response)

        print(first_region_end_index)
        # Perform analysis
        analysis_results = analyzer.analyze_regions(first_region_end_index=first_region_end_index, num_regions=num_regions)
        regions = analysis_results

        
        # Prepare response
        response = {
            'success': True,
            'metadata': {
                'total_data_points': len(analyzer.data),
                'time_column': time_column,
                'response': response,
                'num_regions': num_regions,
                'processing_timestamp': datetime.now().isoformat()
            },
            'regions': regions,
            'analysis': analysis_results
        }
        
        logger.info("Analysis completed successfully")
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error occurred'}), 500

if __name__ == '__main__':
    logger.info("Starting CSV Time Series Analysis Backend")
    app.run(debug=True, host='127.0.0.1', port=5000)