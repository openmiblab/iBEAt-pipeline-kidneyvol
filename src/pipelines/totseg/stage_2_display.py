import os

import numpy as np
from tqdm import tqdm
from totalsegmentator.map_to_binary import class_map
import dbdicom as db

from utils import plot, data
from utils.constants import SITE_IDS




def WIP_movie(build_path, group, site=None,
        kidney=False, all=False, liver_pancreas=False):
    
    datapath = os.path.join(build_path, 'dixon', 'stage_2_data')
    maskpath = os.path.join(build_path, 'totseg', 'stage_1_segment')
    displaypath = os.path.join(build_path, 'totseg', 'stage_2_display')
    os.makedirs(displaypath, exist_ok=True)

    
    if site is None:
        sitedatapath = os.path.join(datapath, group)
        sitemaskpath = os.path.join(maskpath, group)
        sitedisplaypath = os.path.join(displaypath, group)
    else:
        sitedatapath = os.path.join(datapath, site, group)
        sitemaskpath = os.path.join(maskpath, group, site)
        sitedisplaypath = os.path.join(displaypath, group, site)

    # Build output folders
    movies_all = os.path.join(displaypath, sitedisplaypath, 'Movies_all')
    movies_kidneys = os.path.join(displaypath, sitedisplaypath, 'Movies_kidneys')
    movies_liver_pancreas = os.path.join(displaypath, sitedisplaypath, 'Movies_liver_pancreas')
    os.makedirs(movies_all, exist_ok=True)
    os.makedirs(movies_kidneys, exist_ok=True)
    os.makedirs(movies_liver_pancreas, exist_ok=True)

    record = data.dixon_record()

    # Loop over the masks
    for mask in tqdm(db.series(sitemaskpath), 'Displaying masks..'):

        # Get the corresponding outphase series
        patient_id = mask[1]
        study = mask[2][0]
        sequence = data.dixon_series_desc(record, patient_id, study)
        series_op = [sitedatapath, patient_id, study, f'{sequence}_out_phase']

        # Skip if not in the right site
        if patient_id[:4] not in SITE_IDS[site]:
            continue

        # Get arrays
        op_arr = db.volume(series_op).values
        mask_arr = db.volume(mask).values
        rois = {}
        for idx, roi in class_map['total_mr'].items():
            rois[roi] = (mask_arr==idx).astype(np.int16)

        # Build movie (kidneys only)
        if kidney:
            file = os.path.join(movies_kidneys, f'{patient_id}_{study}_{sequence}_kidneys.mp4')
            if not os.path.exists(file):
                rois_k = {k:v for k, v in rois.items() if k in ["kidney_left", "kidney_right"]}
                plot.movie_overlay(op_arr, rois_k, file)

        # Build movie (all ROIS)
        if all:
            file = os.path.join(movies_all, f'{patient_id}_{study}_{sequence}_all.mp4')
            if not os.path.exists(file):
                plot.movie_overlay(op_arr, rois, file)

        if liver_pancreas:
            file = os.path.join(movies_liver_pancreas, f'{patient_id}_{study}_{sequence}_pancreas_liver.mp4')
            if not os.path.exists(file):
                rois_pl = {k:v for k, v in rois.items() if k in ["pancreas", "liver"]}
                plot.movie_overlay(op_arr, rois_pl, file)


def mosaic(build_path, group, site=None, organs=None):

    datapath = os.path.join(build_path, 'dixon', 'stage_2_data')
    maskpath = os.path.join(build_path, 'totseg', 'stage_1_segment')
    displaypath = os.path.join(build_path, 'totseg', 'stage_2_display')
    os.makedirs(displaypath, exist_ok=True)
    
    if site is None:
        sitedatapath = os.path.join(datapath, group)
        sitemaskpath = os.path.join(maskpath, group)
        sitedisplaypath = os.path.join(displaypath, group)
    else:
        sitedatapath = os.path.join(datapath, group, site)
        sitemaskpath = os.path.join(maskpath, group, site)
        sitedisplaypath = os.path.join(displaypath, group, site)

    # Build output folders
    if organs is None:
        sitedisplaypath = os.path.join(sitedisplaypath, 'mosaic_all')
    else:
        sitedisplaypath = os.path.join(sitedisplaypath, 'mosaic_' + '_'.join(organs))
    os.makedirs(sitedisplaypath, exist_ok=True)

    record = data.dixon_record()

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

        # Skip if file already exists
        png_file = os.path.join(sitedisplaypath, f'{patient_id}_{study}_{sequence}.png')
        if os.path.exists(png_file):
             continue

        # Read arrays
        op_arr = db.volume(series_op).values
        mask_arr = db.volume(mask).values
        rois = {}
        for idx, roi in class_map['total_mr'].items():
            rois[roi] = (mask_arr==idx).astype(np.int16)

        # Build mosaic
        if organs is None:
            plot.mosaic_overlay(op_arr, rois, png_file)
        else:
            rois_k = {k:v for k, v in rois.items() if k in organs}
            if rois_k == {}:
                raise ValueError(f'No organs {organs} found in {patient_id} {study}.')
            plot.mosaic_overlay(op_arr, rois_k, png_file)
