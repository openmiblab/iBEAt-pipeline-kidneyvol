import os
import logging

import numpy as np
from tqdm import tqdm
from totalsegmentator.map_to_binary import class_map
import dbdicom as db

from utils import plot


datapath = os.path.join(os.getcwd(), 'build', 'dixon_2_data')
maskpath = os.path.join(os.getcwd(), 'build', 'vat_1_segment')
displaypath = os.path.join(os.getcwd(), 'build', 'vat_2_display')
os.makedirs(displaypath, exist_ok=True)

# Set up logging
logging.basicConfig(
    filename=os.path.join(displaypath, 'error.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)




def overlay(sitedatapath, sitemaskpath, sitedisplaypath, 
            all=False, subcutaneous=False, movie=False, 
            mosaic=False):

    # Build output folders
    display_all = os.path.join(displaypath, sitedisplaypath, 'Mosaic_all')
    display_subcutaneous = os.path.join(displaypath, sitedisplaypath, 'Mosaic_subcutaneous')
    os.makedirs(display_all, exist_ok=True)
    os.makedirs(display_subcutaneous, exist_ok=True)
    movies_all = os.path.join(displaypath, sitedisplaypath, 'Movies_all')
    movies_subcutaneous = os.path.join(displaypath, sitedisplaypath, 'Movies_subcutaneous')
    os.makedirs(movies_all, exist_ok=True)
    os.makedirs(movies_subcutaneous, exist_ok=True)


    # Loop over the masks
    for mask in tqdm(db.series(sitemaskpath), 'Displaying masks..'):

        # Get the corresponding outphase series
        patient_id = mask[1]
        mask_dixon = mask[-1][0]

        # Opposed phase series
        series_op = [sitedatapath, patient_id, 'Baseline', f'{mask_dixon}_out_phase']

        # Get arrays
        op_arr = db.volume(series_op).values
        mask_arr = db.volume(mask).values
        rois = {}
        for idx, roi in class_map['tissue_types_mr'].items():
            rois[roi] = (mask_arr==idx).astype(np.int16)

        # Build movie (subcutaneous only)
        if subcutaneous and movie:
            file = os.path.join(movies_subcutaneous, f'{patient_id}_{mask_dixon}_subcutaneous.mp4')
            if not os.path.exists(file):
                rois_k = {k:v for k, v in rois.items() if k in ["subcutaneous_fat"]}
                plot.movie_overlay(op_arr, rois_k, file)

        # Build movie (all tissues)
        if all and movie:
            file = os.path.join(movies_all, f'{patient_id}_{mask_dixon}_all.mp4')
            if not os.path.exists(file):
                plot.movie_overlay(op_arr, rois, file)
        
        # Build mosaic (subcutaneous only)
        if subcutaneous and mosaic:
            png_file = os.path.join(display_subcutaneous, f'{patient_id}_{mask_dixon}_subcutaneous.png')
            if not os.path.exists(png_file):
                rois_k = {k:v for k, v in rois.items() if k in ["subcutaneous_fat"]}
                plot.mosaic_overlay(op_arr, rois_k, png_file)

        # Build mosaic (all tissues)
        if all and mosaic:
            png_file = os.path.join(display_all, f'{patient_id}_{mask_dixon}_all.png')
            if not os.path.exists(png_file):
                rois = {k:rois[k] for k in ['torso_fat', 'subcutaneous_fat', 'skeletal_muscle']}
                plot.mosaic_overlay(op_arr, rois, png_file)


def leeds():
    sitedatapath = os.path.join(datapath, "BEAt-DKD-WP4-Leeds", "Leeds_Patients") 
    sitemaskpath = os.path.join(maskpath, "BEAt-DKD-WP4-Leeds", "Leeds_Patients")
    sitedisplaypath = os.path.join(displaypath, "BEAt-DKD-WP4-Leeds", "Leeds_Patients")
    overlay(sitedatapath, sitemaskpath, sitedisplaypath, all=True, mosaic=True)


def bari():
    sitedatapath = os.path.join(datapath, "BEAt-DKD-WP4-Bari", "Bari_Patients") 
    sitemaskpath = os.path.join(maskpath, "BEAt-DKD-WP4-Bari", "Bari_Patients")
    sitedisplaypath = os.path.join(displaypath, "BEAt-DKD-WP4-Bari", "Bari_Patients")
    overlay(sitedatapath, sitemaskpath, sitedisplaypath, all=True, mosaic=True)


def all():
    bari()
    leeds()

if __name__=='__main__':
    bari()
    leeds()