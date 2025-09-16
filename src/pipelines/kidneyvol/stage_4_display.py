import os
import logging

import numpy as np
from tqdm import tqdm
import dbdicom as db

from utils import plot, data


datapath = os.path.join(os.getcwd(), 'build', 'dixon', 'stage_2_data')
automaskpath = os.path.join(os.getcwd(), 'build', 'kidneyvol', 'stage_1_segment')
editmaskpath = os.path.join(os.getcwd(), 'build', 'kidneyvol', 'stage_3_edit')
displaypath = os.path.join(os.getcwd(), 'build', 'kidneyvol', 'stage_4_display')
os.makedirs(displaypath, exist_ok=True)


# Set up logging
logging.basicConfig(
    filename=os.path.join(displaypath, 'error.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def movie(sitedatapath, sitemaskpath, sitedisplaypath):

    # Build output folders
    movies_kidneys = os.path.join(displaypath, sitedisplaypath, 'Movies')
    os.makedirs(movies_kidneys, exist_ok=True)

    record = data.dixon_record()
    class_map = {1: "kidney_left", 2: "kidney_right"}

    # Loop over the masks
    for mask in tqdm(db.series(sitemaskpath), 'Displaying masks..'):

        # Get the corresponding outphase series
        patient_id = mask[1]
        study = mask[2][0]
        sequence = data.dixon_series_desc(record, patient_id, study)
        series_op = [sitedatapath, patient_id, study, f'{sequence}_out_phase']

        # Skip if file exists
        file = os.path.join(movies_kidneys, f'{patient_id}_{sequence}_kidneys.mp4')
        if not os.path.exists(file):
            continue

        # Get arrays
        op_arr = db.volume(series_op).values
        mask_arr = db.volume(mask).values
        rois = {}
        for idx, roi in class_map.items():
            rois[roi] = (mask_arr==idx).astype(np.int16)

        # Build movie (kidneys only)
        try:
            plot.movie_overlay(op_arr, rois, file)
        except Exception as e:
            logging.error(f"{patient_id} {sequence} error building movie: {e}")


def mosaic(group, site=None):
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, "Controls") 
        siteautomaskpath = os.path.join(automaskpath, "Controls")
        siteeditmaskpath = os.path.join(editmaskpath, "Controls")
        sitedisplaypath = os.path.join(displaypath, "Controls")
    else:
        sitedatapath = os.path.join(datapath, "Patients", site) 
        siteautomaskpath = os.path.join(automaskpath, "Patients", site)
        siteeditmaskpath = os.path.join(editmaskpath, "Patients", site)
        sitedisplaypath = os.path.join(displaypath, "Patients", site)
    os.makedirs(sitedisplaypath, exist_ok=True)

    record = data.dixon_record()
    class_map = {1: "kidney_left", 2: "kidney_right"}
    all_editmasks = db.series(siteeditmaskpath)

    # Loop over the masks
    for automask in tqdm(db.series(siteautomaskpath), 'Displaying masks..'):

        # Get the corresponding outphase series
        patient_id = automask[1]
        study = automask[2][0]
        sequence = data.dixon_series_desc(record, patient_id, study)
        series_op = [sitedatapath, patient_id, study, f'{sequence}_out_phase']

        # Skip if file exists
        png_file = os.path.join(sitedisplaypath, f'{patient_id}_{study}_{sequence}.png')
        if os.path.exists(png_file):
             continue
        
        # Get image array
        op_arr = db.volume(series_op).values
        
        # Get mask (edited if it exists, else automated)
        editmask = [siteeditmaskpath, patient_id, (study, 0), ('kidney_masks', 0)]
        if editmask in all_editmasks:
            mask_arr = db.volume(editmask).values
        else:
            mask_arr = db.volume(automask).values

        # Get masks
        rois = {}
        for idx, roi in class_map.items():
            rois[roi] = (mask_arr==idx).astype(np.int16)

        # Build mosaic
        try:
            plot.mosaic_overlay(op_arr, rois, png_file)
        except Exception as e:
            logging.error(f"{patient_id} {sequence} error building mosaic: {e}")
