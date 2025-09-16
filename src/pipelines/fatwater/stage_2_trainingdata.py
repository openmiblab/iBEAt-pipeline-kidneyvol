"""
Compute water-dominance masks from data that have fat and water maps
"""

import os
import logging
import json

from tqdm import tqdm
import dbdicom as db

def build_json(num_training):

    trainingdatapath = os.path.join(os.getcwd(), 'build', 'fatwater', 'stage_2_trainingdata', 'nnUNet_raw')
    database = os.path.join(trainingdatapath, "Dataset011_iBEAtFatWater")

    # Build dataset.json file
    json_data = { 
        "channel_names": {  
            "0": "out_phase", 
            "1": "in_phase"
        }, 
        "labels": { 
            "background": 0,
            "water_dominant": 1
        }, 
        "numTraining": num_training, 
        "file_ending": ".nii.gz"
    }
    json_file = os.path.join(database, 'dataset.json')
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)


def generate():

    # Data and results paths
    datapath = os.path.join(os.getcwd(), 'build', 'dixon', 'stage_2_data') 
    waterdompath = os.path.join(os.getcwd(), 'build', 'fatwater', 'stage_1_waterdom') 
    trainingdatapath = os.path.join(os.getcwd(), 'build', 'fatwater', 'stage_2_trainingdata', 'nnUNet_raw')
    os.makedirs(trainingdatapath, exist_ok=True)

    # Create the database folder structure
    database = os.path.join(trainingdatapath, "Dataset011_iBEAtFatWater")
    images_tr = os.path.join(database, 'imagesTr')
    labels_tr = os.path.join(database, 'labelsTr')
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename=os.path.join(trainingdatapath, 'error.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Get water series
    series = db.series(datapath)
    series_water = [s for s in series if s[3][0][-5:]=='water']

    # Loop over the fat series
    num_training = 0
    for series_wi in tqdm(series_water, desc='Writing training data'):

        # Patient and output study
        patient = series_wi[1]
        study = series_wi[2][0]
        series_wi_desc = series_wi[3][0]
        sequence = series_wi_desc[:-6] # remove '_water' suffix

        # Get out_phase/in_phase series and water dominant mask
        series_op = series_wi[:3] + [(f'{sequence}_out_phase', 0)]
        series_ip = series_wi[:3] + [(f'{sequence}_in_phase', 0)]
        series_mask = [waterdompath, patient, (study, 0), (f'{sequence}_water_dominant', 0)]

        # Define the file names of the niftis
        case_id = f"{patient}_{study}_{sequence}"
        file_op = os.path.join(images_tr, f"{case_id}_0000.nii.gz")
        file_ip = os.path.join(images_tr, f"{case_id}_0001.nii.gz")
        file_mask = os.path.join(labels_tr, f"{case_id}.nii.gz")

        # Continue if the case has already been written
        if os.path.exists(file_mask):
            continue

        # Save the inphase/outphase volumes, and the corresponding mask, as niftis
        try:
            db.to_nifti(series_op, file_op, verbose=0)
        except Exception as e:
            logging.error(f"Case{case_id}, out_phase: {e}\n")
            continue
        try:
            db.to_nifti(series_ip, file_ip, verbose=0)
        except Exception as e:
            logging.error(f"Case{case_id}, in_phase: {e}\n")
            continue
        try:
            db.to_nifti(series_mask, file_mask, verbose=0) 
        except Exception as e:
            logging.error(f"Case{case_id}, water_dominant: {e}\n")
            continue
        num_training += 1

    build_json(num_training)


