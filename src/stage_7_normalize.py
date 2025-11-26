import os
import logging

from tqdm import tqdm
import numpy as np
import dbdicom as db
import vreg

from utils import normalize


# Configure logging once, at the start of your script
logging.basicConfig(
    filename='parametrize.log',      # log file name
    level=logging.INFO,           # log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # log format
)


def normalize_kidneys(build_path):

    datapath = os.path.join(build_path, 'kidneyvol', 'stage_3_edit')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')

    kidney_labels = db.series(datapath)
    current_results = db.series(resultspath)

    spacing_norm = 1.0
    volume_norm = 1e6

    for kidney_label in tqdm(kidney_labels, desc=f'Normalizing kidneys..'):
        study = [resultspath] + kidney_label[1:3]

        # Define outputs
        rk_series = study + [('normalized_right_kidney_mask', 0)]
        lk_series = study + [('normalized_left_kidney_mask', 0)]

        # Read data
        if (rk_series not in current_results) or (lk_series not in current_results):
            try:
                vol = db.volume(kidney_label, verbose=0)
            except Exception as e:
                logging.error(f"Cannot read data of patient {rk_series[1]} in study {rk_series[2][0]}: {e}")
                continue

        # Compute right kidney
        if rk_series not in current_results:
            try:
                rk_mask = vol.values == 2
                rk_mask_norm, _ = normalize.normalize_kidney_mask(rk_mask, vol.spacing, 'right')
                rk_vol_norm = vreg.volume(rk_mask_norm.astype(int))
                db.write_volume(rk_vol_norm, rk_series, ref=kidney_label, verbose=0)
            except Exception as e:
                logging.error(f"Error normalizing right kidney of patient {rk_series[1]} in study {rk_series[2][0]}: {e}")

        # Compute left kidney
        if lk_series not in current_results:
            try:
                lk_mask = vol.values == 1
                lk_mask_norm, _ = normalize.normalize_kidney_mask(lk_mask, vol.spacing, 'left')
                lk_vol_norm = vreg.volume(lk_mask_norm.astype(int))
                db.write_volume(lk_vol_norm, lk_series, ref=kidney_label, verbose=0)
            except Exception as e:
                logging.error(f"Error normalizing left kidney of patient {lk_series[1]} in study {lk_series[2][0]}: {e}")



def save_normalized_as_npz(data, build):

    kidney_masks = db.series(data)
    for kidney_mask in tqdm(kidney_masks, desc='Saving kidneys as npz..'):
        patient = kidney_mask[1]
        study_desc, study_id = kidney_mask[2][0], kidney_mask[2][1]
        series_desc, series_nr = kidney_mask[3][0], kidney_mask[3][1]
        filedir = os.path.join(
            build, 
            f"Patient__{patient}", 
            f"Study__{study_id + 1}__{study_desc}",
        )
        os.makedirs(filedir, exist_ok=True)
        filename = f"Series__{series_nr + 1}__{series_desc}.npz"
        filepath = os.path.join(filedir, filename)
        if os.path.exists(filepath):
            continue
        vol = db.volume(kidney_mask, verbose=0)
        array = vol.values.astype(bool)
        np.savez_compressed(filepath, mask=array)




if __name__ == '__main__':

    DATA_DIR = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\kidneyvol\stage_7_normalized"
    RESULTS_DIR = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\kidneyvol\stage_7_normalized_npz"

    # save_normalized_as_npz(DATA_DIR, RESULTS_DIR)
