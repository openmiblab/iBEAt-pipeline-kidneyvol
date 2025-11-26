import os
import logging
import json
import argparse
import time
from pathlib import Path

from tqdm import tqdm
import numpy as np
import dask

import utils.sdf
import utils.lb
import utils.data


def _spectral_feature_vector(kidney_mask, k0, resultspath):

    # Define filepath
    label, value = utils.data.parse_npz_dbfile(kidney_mask)
    filepath = os.path.join(resultspath, f"{label}.npz")
    if os.path.exists(filepath):
        return

    # Compute coeffs
    mask_norm = np.load(kidney_mask)['mask']
    sdf = utils.sdf.coeffs_from_mask(mask_norm)
    coeffs = sdf[:k0, :k0, :k0]

    # Save with label
    np.savez_compressed(filepath, coeffs=coeffs, **value)


def build_spectral_feature_vectors(project_path, debug=False):

    datapath = os.path.join(project_path, "stage_7_normalized_npz")
    resultspath = os.path.join(project_path, "stage_8_features_sdf")

    k0 = 32

    logging.info("Computation of correlation matrices started")
    t0 = time.time()

    os.makedirs(resultspath, exist_ok=True)

    # Get all baseline kidneys
    kidney_masks = [f for f in Path(datapath).rglob("*") if f.is_file()]

    if debug:
        kidney_masks = kidney_masks[:20]

    # Compute feature vectors
    tasks = [dask.delayed(_spectral_feature_vector)(mask, k0, resultspath) for mask in kidney_masks]
    dask.compute(*tasks)

    t1 = time.time()
    print(f"Computation time: {t1 - t0:.3f} seconds")
    logging.info("Computation of correlation matrices completed.")



def _lb_feature_vector(kidney_mask, lb_cutoff):

    # Get info
    labels = {}
    label, value = utils.data.parse_npz_dbfile(kidney_mask)
    labels[label] = value

    # Compute coeffs
    coeffs = {}
    mask_norm = np.load(kidney_mask)['mask']
    coeffs[label] = utils.lb.eigvals(mask_norm, k=lb_cutoff)

    return coeffs, labels


def build_lb_feature_vectors(project_path, debug=False):

    datapath = os.path.join(project_path, "stage_7_normalized_npz")
    resultspath = os.path.join(project_path, "stage_8_features_lb")

    lb_cutoff = 100

    logging.info("Computation of correlation matrices started")
    t0 = time.time()

    os.makedirs(resultspath, exist_ok=True)

    # Get all baseline kidneys
    kidney_masks = [f for f in Path(datapath).rglob("*") if f.is_file()]

    if debug:
        kidney_masks = kidney_masks[:20] 

    # Compute feature vectors
    tasks = [dask.delayed(_lb_feature_vector)(mask, lb_cutoff) for mask in kidney_masks]
    results = dask.compute(*tasks)

    # Save results
    filename = f'normalized_kidney_lb_eigvals_{lb_cutoff}'

    # Save feature vectors
    coeffs = {k: v for d in results for k, v in d[0].items()}
    filepath = os.path.join(resultspath, f'{filename}.npz')
    np.savez_compressed(filepath, **coeffs)

    # Save labels
    labels = {k: v for d in results for k, v in d[1].items()}
    filepath = os.path.join(resultspath, f'{filename}.json')
    with open(filepath, 'w', encoding='utf-8') as fp:
        json.dump(labels, fp, indent=4)

    t1 = time.time()
    print(f"Computation time: {t1 - t0:.3f} seconds")
    logging.info("Computation of correlation matrices completed.")



if __name__ == '__main__':

    DATA = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\kidneyvol"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA, help="Project folder")
    parser.add_argument("--debug", action="store_true",  help="Debug")
    args = parser.parse_args()

    build_spectral_feature_vectors(args.data, args.debug)
    #build_lb_feature_vectors(args.proj, args.debug)