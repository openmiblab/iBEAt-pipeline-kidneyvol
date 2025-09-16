"""
Compute water-dominance masks from data that have fat and water maps
"""

import os
import logging

from tqdm import tqdm
import numpy as np
import dbdicom as db



def compute(group, site=None):

    # Define global paths
    datapath = os.path.join(os.getcwd(), 'build', 'dixon', 'stage_2_data') 
    waterdompath = os.path.join(os.getcwd(), 'build', 'fatwater', 'stage_1_waterdom') 
    os.makedirs(waterdompath, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename=os.path.join(waterdompath, 'error.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Define site paths
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, group) 
        sitewaterdompath = os.path.join(waterdompath, group)
    else:
        sitedatapath = os.path.join(datapath, group, site)
        sitewaterdompath = os.path.join(waterdompath, group, site)
    os.makedirs(sitewaterdompath, exist_ok=True)

    # Get all water series
    series = db.series(sitedatapath)
    series_water = [s for s in series if s[3][0][-5:]=='water']

    # Loop over the fat series
    existing_series = db.series(sitewaterdompath)
    for series_wi in tqdm(series_water, desc='Computing water-dominant masks'):

        # Patient and output study
        patient = series_wi[1]
        study = series_wi[2][0]
        series_wi_desc = series_wi[3][0]
        sequence = series_wi_desc[:-6] # remove '_water' suffix

        # Skip if the water dominant map already already exists
        waterdom_series = [sitewaterdompath, patient, (study, 0), (f'{sequence}_water_dominant', 0)]
        if waterdom_series in existing_series:
            continue

        # Get corresponding fat series
        series_fi = series_wi[:3] + [(sequence + '_fat', 0)]

        # Read the fat and water volumes
        try:
            wi = db.volume(series_wi)
            fi = db.volume(series_fi)
        except Exception as e:
            logging.error(f"Patient {patient} - error reading F-W {sequence}: {e}")
            continue

        # Compute water-dominant mask
        try:
            # Build label array (0=Air, 1=Water dominant, 2=Fat dominant)
            label_array = np.zeros(wi.values.shape, dtype=np.int16)
            label_array[wi.values > fi.values] = 1
            #label_array[fat_dominant & foreground] = 2
        except Exception as e:
            logging.error(f"Error computing water-dominant mask for {patient} {sequence}: {e}")
            continue
        
        db.write_volume((label_array, wi.affine), waterdom_series, ref=series_wi)

