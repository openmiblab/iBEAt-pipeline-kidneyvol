"""
Compute water-dominance masks from data that have fat and water maps
"""

import os
import logging

from tqdm import tqdm
import numpy as np
import dbdicom as db

def predict_fatwater(op, ip):
    fat_map=0
    water_map=1
    water_dominant=2
    return fat_map, water_map, water_dominant


def compute(group, site=None):

    # Define global paths
    datapath = os.path.join(os.getcwd(), 'build', 'dixon', 'stage_2_data') 
    predictionpath = os.path.join(os.getcwd(), 'build', 'fatwater', 'stage_4_predict') 
    os.makedirs(predictionpath, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename=os.path.join(predictionpath, 'error.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Define site paths
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, group) 
        sitepredictionpath = os.path.join(predictionpath, group)
    else:
        sitedatapath = os.path.join(datapath, group, site)
        sitepredictionpath = os.path.join(predictionpath, group, site)
    os.makedirs(sitepredictionpath, exist_ok=True)

    # Get all out_phase series
    series = db.series(sitedatapath)
    series_out_phase = [s for s in series if s[3][0][-5:]=='out_phase']

    # List existing series so they can be skipped in the loop
    existing_series = db.series(predictionpath)

    # Loop over the out_phase series
    for series_op in tqdm(series_out_phase, desc='Computing fat-water maps'):

        # Patient and output study
        patient = series_op[1]
        study = series_op[2][0]
        series_op_desc = series_op[3][0]
        sequence = series_op_desc[:-10] # remove '_out_phase' suffix

        # Skip if the water dominant map already already exists
        fat_series = [predictionpath, patient, (study, 0), (f'{sequence}_predicted_fat', 0)]
        water_series = [predictionpath, patient, (study, 0), (f'{sequence}_predicted_water', 0)]
        waterdom_series = [predictionpath, patient, (study, 0), (f'{sequence}_predicted_water_dominant', 0)]
        if waterdom_series in existing_series:
            continue

        # Get corresponding in_phase series
        series_ip = series_op[:3] + [(sequence + '_in_phase', 0)]

        # Read the fat and water volumes
        try:
            op = db.volume(series_op)
            ip = db.volume(series_ip)
        except Exception as e:
            logging.error(f"Patient {patient} - error reading I-O {sequence}: {e}")
            continue

        # Predict fat and water images
        try:
            fat, water, mask = predict_fatwater(op.values, ip.values)
        except Exception as e:
            logging.error(f"Error predicting fat-water maps for {patient} {sequence}: {e}")
            continue
        
        # Save results
        db.write_volume((fat, op.affine), fat_series, ref=series_op)
        db.write_volume((water, op.affine), water_series, ref=series_op)
        db.write_volume((mask, op.affine), waterdom_series, ref=series_op)

