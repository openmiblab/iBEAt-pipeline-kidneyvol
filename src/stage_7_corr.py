import os
import logging
import argparse

from tqdm import tqdm
import numpy as np
import dbdicom as db
import pandas as pd

from utils import sdf, lb, normalize


# Configure logging once, at the start of your script
logging.basicConfig(
    filename='log_corr.log',      # log file name
    level=logging.INFO,           # log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # log format
)


def build_all_correlation_matrices(datapath, resultspath):

    logging.info("Computation of correlation matrices started")

    os.makedirs(resultspath, exist_ok=True)

    # Get baseline kidneys
    kidney_masks = db.series(datapath)
    kidney_masks = [k for k in kidney_masks if k[2][0] in ['Visit1', 'Baseline']]
    # kidney_masks = kidney_masks[:3] # for debugging

    n = len(kidney_masks)
    dice = np.zeros((n, n))
    cov = np.zeros((n, n))
    cov_sdf = np.zeros((n, n))
    cov_lb = np.zeros((n, n))
    sdf_cutoff = (32, 32, 32)
    lb_cutoff = 100
    labels = [''] * n

    pbar = tqdm(total=int(n*(n+1)/2), desc="Computing correlations")
    cnt = 0
    for i, kidney_mask_i in enumerate(kidney_masks):

        # Create label
        patient = kidney_mask_i[1]
        kidney = 'left' if 'left' in kidney_mask_i[3][0] else 'right'
        labels[i] = f"{patient}_{kidney}"

        # Compute DICE
        mask_norm_i = db.volume(kidney_mask_i, verbose=0).values
        sdf_i = sdf.coeffs_from_mask(mask_norm_i, sdf_cutoff, normalize=True)
        lb_i = lb.eigvals(mask_norm_i, k=lb_cutoff, normalize=True)

        for j, kidney_mask_j in enumerate(kidney_masks[i:]):

            pbar.update(1)
            cnt += 1
            logging.info(f"Processing correlation {cnt}/{n*(n+1)/2}")

            mask_norm_j = db.volume(kidney_mask_j, verbose=0).values
            sdf_j = sdf.coeffs_from_mask(mask_norm_j, sdf_cutoff, normalize=True)
            lb_j = lb.eigvals(mask_norm_j, k=lb_cutoff, normalize=True)

            # dice[i, i+j] = normalize.dice_coefficient(mask_norm_i, mask_norm_j)
            dice[i, i+j] = normalize.advance_dice_coefficient(mask_norm_i, mask_norm_j,axis=2,return_angle = False,return_mask=False)
            cov[i, i+j] = normalize.covariance(mask_norm_i, mask_norm_j)
            cov_sdf[i, i+j] = normalize.covariance(sdf_i, sdf_j)
            cov_lb[i, i+j] = normalize.covariance(lb_i, lb_j)

            dice[i+j, i] = dice[i, i+j]
            cov[i+j, i] = cov[i, i+j]
            cov_sdf[i+j, i] = cov_sdf[i, i+j]
            cov_lb[i+j, i] = cov_lb[i, i+j]

            # Save results as csv
            file = os.path.join(resultspath, f'normalized_kidney_dice.csv')
            df = pd.DataFrame(dice, columns=labels, index=labels)
            df.to_csv(file)

            file = os.path.join(resultspath, f'normalized_kidney_cov.csv')
            df = pd.DataFrame(cov, columns=labels, index=labels)
            df.to_csv(file)

            file = os.path.join(resultspath, f'normalized_kidney_cov_sdf.csv')
            df = pd.DataFrame(cov_sdf, columns=labels, index=labels)
            df.to_csv(file)

            file = os.path.join(resultspath, f'normalized_kidney_cov_lb.csv')
            df = pd.DataFrame(cov_lb, columns=labels, index=labels)
            df.to_csv(file)


    logging.info("Computation of correlation matrices completed.")



if __name__ == '__main__':

    DATA_DIR = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\kidneyvol\stage_7_normalized"
    RESULTS_DIR = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\kidneyvol\stage_7_shape_analysis"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA_DIR, help="Data folder")
    parser.add_argument("--build", type=str, default=RESULTS_DIR, help="Build folder")
    args = parser.parse_args()

    build_all_correlation_matrices(args.data, args.build)
