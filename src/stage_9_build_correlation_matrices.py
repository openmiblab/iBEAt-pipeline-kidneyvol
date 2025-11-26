import os
import logging
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import dask

# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

from utils import normalize
import utils.data



# def principal_component_analysis(build_path):

#     n_components = 10
#     n_clusters = 3
    
#     resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_shape_analysis')

#     for kidney in ['left', 'right']:
#         df_file = os.path.join(resultspath, f'{kidney}_kidney_features.csv')
#         df_features = pd.read_csv(df_file)

#         # Run PCA on feature matrix
#         pca = PCA(n_components=n_components)
#         features_reduced = pca.fit_transform(df_features.values)

#         # Cluster shapes in PCA space
#         kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
#         labels = kmeans.fit_predict(features_reduced)

#         # Reconstruct shapes for the different features
#         # coeffs = features_reduced[0].reshape((32,32,32)) 
#         # mask_recon = sdf.mask_from_coeffs(coeffs)


def _correlation_matrix(kidney_masks, featurepath, resultspath, i, j):

    # Define path
    tmp_path = os.path.join(resultspath, 'tmp')
    os.makedirs(tmp_path, exist_ok=True)
    filepath = os.path.join(tmp_path, f"cov_{i}_{j}.npz")
    if os.path.exists(filepath):
        return

    # Get labels
    label_i, _ = utils.data.parse_npz_dbfile(kidney_masks[i])
    label_j, _ = utils.data.parse_npz_dbfile(kidney_masks[j])

    # Feature paths
    sdf_file_i = os.path.join(featurepath, f"{label_i}.npz")
    sdf_file_j = os.path.join(featurepath, f"{label_j}.npz")

    # Load masks
    mask_norm_i = np.load(kidney_masks[i])['mask']
    mask_norm_j = np.load(kidney_masks[j])['mask']

    # Load features
    sdf_i = np.load(sdf_file_i)['coeffs'].flatten()
    sdf_j = np.load(sdf_file_j)['coeffs'].flatten()
    
    # Compute cov
    result = {
        'dice': normalize.dice_coefficient(mask_norm_i, mask_norm_j),
        'cov': normalize.covariance(mask_norm_i, mask_norm_j),
        'cov_sdf': normalize.covariance(sdf_i, sdf_j),
    }

    # Save to disk
    np.savez_compressed(filepath, **result)


def _baseline_kidneys(datapath):
    kidney_masks = []
    labels = []
    for f in Path(datapath).rglob("*"):
        if f.is_file():
            label, value = utils.data.parse_npz_dbfile(f)
            if value['Study'] in ['Visit1', 'Baseline']:
                kidney_masks.append(f)
                labels.append(label)
    return kidney_masks, labels


def compute_correlations(project_path, debug=False):

    datapath = os.path.join(project_path, "stage_7_normalized_npz")
    featurepath = os.path.join(project_path, "stage_8_features_sdf")
    resultspath = os.path.join(project_path, "stage_9_shape_analysis")

    logging.info("Computation of correlation matrices started")
    t0 = time.time()

    # Get all baseline kidneys
    kidney_masks, labels = _baseline_kidneys(datapath)

    if debug:
        kidney_masks = kidney_masks[:20]
        labels = labels[:20]

    # Build output
    n = len(kidney_masks)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]

    # Load sdf features
    tasks = [dask.delayed(_correlation_matrix)(kidney_masks, featurepath, resultspath, i, j) for i,j in pairs]
    dask.compute(*tasks)

    t1 = time.time()
    print(f"Computation time: {t1 - t0:.3f} seconds")
    logging.info("Computation of correlation matrices completed.")



def build_correlation_matrices(project_path):

    datapath = os.path.join(project_path, "stage_7_normalized_npz")
    resultspath = os.path.join(project_path, "stage_9_shape_analysis")

    os.makedirs(resultspath, exist_ok=True)

    _, labels = _baseline_kidneys(datapath)

    # Since these are baselines only, remove BL or V1 from the label
    # labels = [s.replace('-BL', '').replace('-V1', '') for s in labels]

    # Build output arrays
    n = len(labels)
    dice_arr = np.zeros((n, n))
    cov_arr = np.zeros((n, n))
    cov_sdf_arr = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):

            filepath = os.path.join(resultspath, 'tmp', f"cov_{i}_{j}.npz")
            vals = np.load(filepath)

            dice_arr[i, j] = vals['dice']
            cov_arr[i, j] = vals['cov']
            cov_sdf_arr[i, j] = vals['cov_sdf']

            dice_arr[j, i] = vals['dice']
            cov_arr[j, i] = vals['cov']
            cov_sdf_arr[j, i] = vals['cov_sdf']

    cov_arr /= cov_arr.max()
    cov_sdf_arr /= cov_sdf_arr.max()

    # Save arrays as csv
    file = os.path.join(resultspath, f'normalized_kidney_dice.csv')
    df = pd.DataFrame(dice_arr, columns=labels, index=labels)
    df.to_csv(file)

    file = os.path.join(resultspath, f'normalized_kidney_cov.csv')
    df = pd.DataFrame(cov_arr, columns=labels, index=labels)
    df.to_csv(file)

    file = os.path.join(resultspath, f'normalized_kidney_cov_sdf.csv')
    df = pd.DataFrame(cov_sdf_arr, columns=labels, index=labels)
    df.to_csv(file)




def check_correlation_matrices(project_path):

    datapath = os.path.join(project_path, "stage_7_normalized_npz")
    resultspath = os.path.join(project_path, "stage_9_shape_analysis")

    # Baseline kidney labels from masks
    _, labels = _baseline_kidneys(datapath)

    files = [
        f'normalized_kidney_dice.csv',
        f'normalized_kidney_cov.csv',
        f'normalized_kidney_cov_sdf.csv',
    ]

    for f in files:
        file = os.path.join(resultspath, f)
        df = pd.read_csv(file, index_col=0)
        msg = f"""{f} is complete."""
        if set(df.columns) > set(labels):
            msg = f"""{f} has too many variables!! 

                The following variables are in the correlation matrix but 
                not in the dataset: {set(df.columns) - set(labels)}
            """
        if set(df.columns) < set(labels):
            msg = f"""{f} has insufficient variables!! 

                The following variables are in the dataset but 
                not in the correlation matrix: {set(labels) - set(df.columns)}
            """
        print(msg)
        logging.info(msg)


if __name__ == '__main__':

    DATA = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\kidneyvol"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA, help="Project folder")
    parser.add_argument("--debug", action="store_true",  help="Debug")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        filename=os.path.join(DATA, 'log.txt'),      # log file name
        level=logging.INFO,           # log level
        format='%(asctime)s - %(levelname)s - %(message)s',  # log format
    )

    #compute_correlations(args.data)
    build_correlation_matrices(args.data, args.debug)
    #check_correlation_matrices(args.data)
