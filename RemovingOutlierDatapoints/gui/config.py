import logging
import os
import shutil # You'll need this for 'overwriting' if that's truly what you mean
from pathlib import Path

# Configure logging (optional, but good practice for INFO messages)
# You might want to configure logging more globally in your app
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

BASE_DIR = Path.home() / "ICP-MS Plots"
OUTPUT_PATH = BASE_DIR / 'output' # Simpler way to join paths with Pathlib
FIGURES_PATH = OUTPUT_PATH / 'figures'
EXPORT_DATA_PATH = OUTPUT_PATH / 'export_data'


# Ensure BASE_DIR exists (it's generally a good idea not to delete BASE_DIR itself)
os.makedirs(BASE_DIR, exist_ok=True)
logging.info(f"Ensured base directory exists: {BASE_DIR}")

if OUTPUT_PATH.exists():
    logging.info(f"Removing existing output directory: {OUTPUT_PATH}")
    shutil.rmtree(OUTPUT_PATH) # Deletes the directory and all its contents

# Now, create all necessary output subdirectories
logging.info(f"Creating output directory: {OUTPUT_PATH}")
os.makedirs(OUTPUT_PATH, exist_ok=True) # Recreate the main output directory

logging.info(f"Creating figures directory: {FIGURES_PATH}")
os.makedirs(FIGURES_PATH, exist_ok=True)

logging.info(f"Creating export data directory: {EXPORT_DATA_PATH}")
os.makedirs(EXPORT_DATA_PATH, exist_ok=True)


logging.info("Configuration Module Complete.")
logging.info(f'OUTPUT_PATH: {OUTPUT_PATH}')
logging.info(f'FIGURES_PATH: {FIGURES_PATH}')
logging.info(f'EXPORT_DATA_PATH: {EXPORT_DATA_PATH}')