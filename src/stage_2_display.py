import os
import logging

import numpy as np
from tqdm import tqdm
import dbdicom as db

from utils import plot, data
from utils.constants import SITE_IDS



def mosaic(build_path, group, site=None):

    datapath = os.path.join(build_path, 'dixon', 'stage_2_data')
    maskpath = os.path.join(build_path, 'kidneyvol', 'stage_1_segment')
    displaypath = os.path.join(build_path, 'kidneyvol', 'stage_2_display')
    os.makedirs(displaypath, exist_ok=True)

    if group == 'Controls':
        sitedatapath = os.path.join(datapath, "Controls") 
        sitemaskpath = os.path.join(maskpath, "Controls")
        sitedisplaypath = os.path.join(displaypath, "Controls")
    else:
        sitedatapath = os.path.join(datapath, "Patients", site) 
        sitemaskpath = os.path.join(maskpath, "Patients", site)
        sitedisplaypath = os.path.join(displaypath, "Patients", site)
    os.makedirs(sitedisplaypath, exist_ok=True)

    record = data.dixon_record()
    class_map = {1: "kidney_left", 2: "kidney_right"}

    # Loop over the masks
    for mask in tqdm(db.series(sitemaskpath), 'Displaying masks..'):

        # Get the corresponding outphase series
        patient_id = mask[1]
        study = mask[2][0]
        sequence = data.dixon_series_desc(record, patient_id, study)
        series_op = [sitedatapath, patient_id, study, f'{sequence}_out_phase']

        # Skip if not in the right site
        if site is not None:
            if patient_id[:4] not in SITE_IDS[site]:
                continue

        # Skip if file exists
        png_file = os.path.join(sitedisplaypath, f'{patient_id}_{study}_{sequence}.png')
        if os.path.exists(png_file):
             continue

        # Get arrays
        op_arr = db.volume(series_op).values
        mask_arr = db.volume(mask).values
        rois = {}
        for idx, roi in class_map.items():
            rois[roi] = (mask_arr==idx).astype(np.int16)

        # Build mosaic
        try:
            plot.mosaic_overlay(op_arr, rois, png_file)
        except Exception as e:
            logging.error(f"{patient_id} {sequence} error building mosaic: {e}")
