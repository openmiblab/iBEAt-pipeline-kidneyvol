import os
import logging
import json

from tqdm import tqdm
import numpy as np
import dbdicom as db
import pyvista as pv
import pandas as pd
import vreg
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from utils import normalize, sdf, constants, lb


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
        

def in_site(patient_id, site):
    for site_id in constants.SITE_IDS[site]:
        if site_id in patient_id:
            return True
    return False

def display_all_normalizations(build_path, group=None, site=None):

    # Paths
    datapath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_shape_analysis')
    os.makedirs(resultspath, exist_ok=True)
    if group is None:
        prefix = ''
    elif group == 'Controls':
        prefix = 'controls_'
    else:
        prefix = f"patients_{site}_"

    # Plot settings
    aspect_ratio = 16/9
    width = 150
    height = 150

    # Get baseline kidneys
    kidney_masks = db.series(datapath)
    kidney_masks = [k for k in kidney_masks if k[2][0] in ['Visit1', 'Baseline']]
    if group == 'Controls':
        kidney_masks = [k for k in kidney_masks if 'C' in k[1]]
    if group == 'Patients':
        kidney_masks = [k for k in kidney_masks if in_site(k[1], site)]

    for kidney in ['right', 'left']:
    #for kidney in ['left']:

        masks = [k for k in kidney_masks if k[3][0] == f'normalized_{kidney}_kidney_mask']

        # Count nr of mosaics
        n_mosaics = len(masks)
        nrows = int(np.round(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
        ncols = int(np.ceil(n_mosaics/nrows))

        plotter = pv.Plotter(window_size=(ncols*width, nrows*height), shape=(nrows, ncols), border=False, off_screen=True)
        plotter.background_color = 'white'

        row = 0
        col = 0
        cnt = 0
        for mask_series in tqdm(masks, desc=f'Processing kidney {kidney}'):

            patient = mask_series[1]

            # Set up plotter
            plotter.subplot(row,col)
            plotter.add_text(f"{patient}", font_size=6)
            if col == ncols-1:
                col = 0
                row += 1
            else:
                col += 1

            # Load data
            vol = db.volume(mask_series, verbose=0)
            mask_norm, _ = sdf.compress(vol.values, (32, 32, 32))

            # Plot tile
            orig_vol = pv.wrap(mask_norm.astype(float))
            orig_vol.spacing = vol.spacing
            orig_surface = orig_vol.contour(isosurfaces=[0.5])
            plotter.add_mesh(orig_surface, color='lightblue', opacity=1.0, style='surface')
            plotter.camera_position = 'iso'
            plotter.view_vector((1, 0, 0))  # rotate 180° around vertical axis

            # cnt+=1
            # if cnt==2: # for debugging
            #     break
        
        imagefile = os.path.join(resultspath, f"{prefix}kidney_{kidney}.png")
        plotter.screenshot(imagefile)
        plotter.close()



def display_subject_clusters(build_path, data_path):

    cluster_ids = os.path.join(data_path, 'subject_cluster_ids.json')
    # Load JSON file back into a Python dictionary
    with open(cluster_ids, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    # JSON keys are strings by default — convert them back to ints if needed:
    clusters = {int(k): v for k, v in clusters.items()}

    # Paths
    datapath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_shape_analysis')
    os.makedirs(resultspath, exist_ok=True)

    # Plot settings
    aspect_ratio = 16/9
    width = 150
    height = 150

    # Get baseline kidneys
    all_masks = db.series(datapath)
    all_masks = [k for k in all_masks if k[2][0] in ['Visit1', 'Baseline']]

    for cluster, cluster_ids in clusters.items():

        # if cluster<=3: # !!!! temporary - got this one already
        #     continue

        # Get masks for the cluster
        cluster_masks = [k for k in all_masks if k[1] in cluster_ids]

        for kidney in ['right', 'left']:

            # if cluster==4 and kidney=='right': # !!!!! temporary - got this one already
            #     continue

            # Get masks for the kidney
            masks = [k for k in cluster_masks if k[3][0] == f'normalized_{kidney}_kidney_mask']

            # Count nr of mosaics
            n_mosaics = len(masks)
            nrows = int(np.round(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
            ncols = int(np.ceil(n_mosaics/nrows))

            plotter = pv.Plotter(window_size=(ncols*width, nrows*height), shape=(nrows, ncols), border=False, off_screen=True)
            plotter.background_color = 'white'

            row = 0
            col = 0
            cnt = 0
            for mask_series in tqdm(masks, desc=f'Processing cluster {cluster}, kidney {kidney}'):

                patient = mask_series[1]

                # Set up plotter
                plotter.subplot(row,col)
                plotter.add_text(f"{patient}", font_size=6)
                if col == ncols-1:
                    col = 0
                    row += 1
                else:
                    col += 1

                # Load data
                vol = db.volume(mask_series, verbose=0)
                mask_norm, _ = sdf.compress(vol.values, (32, 32, 32))

                # Plot tile
                orig_vol = pv.wrap(mask_norm.astype(float))
                orig_vol.spacing = vol.spacing
                orig_surface = orig_vol.contour(isosurfaces=[0.5])
                plotter.add_mesh(orig_surface, color='lightblue', opacity=1.0, style='surface')
                plotter.camera_position = 'iso'
                plotter.view_vector((1, 0, 0))  # rotate 180° around vertical axis

                # cnt+=1
                # if cnt==2: # for debugging
                #     break
            
            imagefile = os.path.join(resultspath, f"cluster_{cluster}_kidney_{kidney}.png")
            plotter.screenshot(imagefile)
            plotter.close()


def display_kidney_clusters(build_path, data_path):

    cluster_ids = os.path.join(data_path, 'kidney_cluster_ids.json')
    # Load JSON file back into a Python dictionary
    with open(cluster_ids, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    # JSON keys are strings by default — convert them back to ints if needed:
    clusters = {int(k): v for k, v in clusters.items()}

    # Paths
    datapath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_shape_analysis')
    os.makedirs(resultspath, exist_ok=True)

    # Plot settings
    aspect_ratio = 16/9
    width = 150
    height = 150

    # Get baseline kidneys
    all_masks = db.series(datapath)
    all_masks = [k for k in all_masks if k[2][0] in ['Visit1', 'Baseline']]

    for cluster, cluster_ids in clusters.items():

        # if cluster==1:
        #     continue

        # Get masks for the cluster
        cluster_masks = []
        for mask in all_masks:
            patient_id, series_desc = mask[1], mask[3][0]
            kidney_id = patient_id + '_L' if 'left' in series_desc else patient_id + '_R'
            if kidney_id in cluster_ids:
                cluster_masks.append(mask)

        # Count nr of mosaics
        n_mosaics = len(cluster_masks)
        nrows = int(np.round(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
        ncols = int(np.ceil(n_mosaics/nrows))

        plotter = pv.Plotter(window_size=(ncols*width, nrows*height), shape=(nrows, ncols), border=False, off_screen=True)
        plotter.background_color = 'white'

        row = 0
        col = 0
        for mask_series in tqdm(cluster_masks, desc=f'Processing cluster {cluster}'):

            patient = mask_series[1]

            # Set up plotter
            plotter.subplot(row,col)
            plotter.add_text(f"{patient}", font_size=6)
            if col == ncols-1:
                col = 0
                row += 1
            else:
                col += 1

            # Load data
            vol = db.volume(mask_series, verbose=0)
            mask_norm, _ = sdf.compress(vol.values, (32, 32, 32))

            # Plot tile
            orig_vol = pv.wrap(mask_norm.astype(float))
            orig_vol.spacing = vol.spacing
            orig_surface = orig_vol.contour(isosurfaces=[0.5])
            plotter.add_mesh(orig_surface, color='lightblue', opacity=1.0, style='surface')
            plotter.camera_position = 'iso'
            plotter.view_vector((1, 0, 0))  # rotate 180° around vertical axis
        
        imagefile = os.path.join(resultspath, f"kidney_cluster_{cluster}.png")
        plotter.screenshot(imagefile)
        plotter.close()


def build_dice_correlation_matrix(build_path):

    datapath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_shape_analysis')

    # Get baseline kidneys
    kidney_masks = db.series(datapath)
    kidney_masks = [k for k in kidney_masks if k[2][0] in ['Visit1', 'Baseline']]
    # kidney_masks = kidney_masks[:2] # !!!! for debugging

    n = len(kidney_masks)
    dice = np.zeros((n, n))
    cov = np.zeros((n, n))
    labels = []

    pbar = tqdm(total=int(n*(n-1)/2), desc="Computing correlations")
    for i, kidney_mask_i in enumerate(kidney_masks):

        # Create label
        patient = kidney_mask_i[1]
        kidney = 'left' if 'left' in kidney_mask_i[3][0] else 'right'
        labels.append(f"{patient}_{kidney}")

        # Compute DICE
        mask_norm_i = db.volume(kidney_mask_i, verbose=0).values

        for j, kidney_mask_j in enumerate(kidney_masks[i:]):

            mask_norm_j = db.volume(kidney_mask_j, verbose=0).values

            dice[i, i+j] = normalize.dice_coefficient(mask_norm_i, mask_norm_j)
            cov[i, i+j] = normalize.covariance(mask_norm_i, mask_norm_j)

            dice[i+j, i] = dice[i, i+j]
            cov[i+j, i] = cov[i, i+j]

            pbar.update(1)

    # Save results as csv
    dice = pd.DataFrame(dice, columns=labels, index=labels)
    file = os.path.join(resultspath, f'normalized_kidney_dice.csv')
    dice.to_csv(file)

    cov = pd.DataFrame(cov, columns=labels, index=labels)
    file = os.path.join(resultspath, f'normalized_kidney_cov.csv')
    cov.to_csv(file)


def build_all_correlation_matrices(build_path):

    datapath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_shape_analysis')

    # Get baseline kidneys
    kidney_masks = db.series(datapath)
    kidney_masks = [k for k in kidney_masks if k[2][0] in ['Visit1', 'Baseline']]
    # kidney_masks = kidney_masks[:2] # !!!! for debugging

    n = len(kidney_masks)
    dice = np.zeros((n, n))
    cov = np.zeros((n, n))
    cov_sdf = np.zeros((n, n))
    cov_lb = np.zeros((n, n))
    sdf_cutoff = (32, 32, 32)
    lb_cutoff = 100
    labels = []

    pbar = tqdm(total=int(n*(n-1)/2), desc="Computing correlations")
    for i, kidney_mask_i in enumerate(kidney_masks):

        # Create label
        patient = kidney_mask_i[1]
        kidney = 'left' if 'left' in kidney_mask_i[3][0] else 'right'
        labels.append(f"{patient}_{kidney}")

        # Compute DICE
        mask_norm_i = db.volume(kidney_mask_i, verbose=0).values
        sdf_i = sdf.coeffs_from_mask(mask_norm_i, sdf_cutoff, normalize=True)
        lb_i = lb.eigvals(mask_norm_i, k=lb_cutoff, normalize=True)

        for j, kidney_mask_j in enumerate(kidney_masks[i:]):

            mask_norm_j = db.volume(kidney_mask_j, verbose=0).values
            sdf_j = sdf.coeffs_from_mask(mask_norm_j, sdf_cutoff, normalize=True)
            lb_j = lb.eigvals(mask_norm_j, k=lb_cutoff, normalize=True)

            dice[i, i+j] = normalize.dice_coefficient(mask_norm_i, mask_norm_j)
            cov[i, i+j] = normalize.covariance(mask_norm_i, mask_norm_j)
            cov_sdf[i, i+j] = normalize.covariance(sdf_i, sdf_j)
            cov_lb[i, i+j] = normalize.covariance(lb_i, lb_j)

            dice[i+j, i] = dice[i, i+j]
            cov[i+j, i] = cov[i, i+j]
            cov_sdf[i+j, i] = cov_sdf[i, i+j]
            cov_lb[i+j, i] = cov_lb[i, i+j]

            pbar.update(1)

    # Save results as csv
    dice = pd.DataFrame(dice, columns=labels, index=labels)
    file = os.path.join(resultspath, f'normalized_kidney_dice.csv')
    dice.to_csv(file)

    cov = pd.DataFrame(cov, columns=labels, index=labels)
    file = os.path.join(resultspath, f'normalized_kidney_cov.csv')
    cov.to_csv(file)

    cov_sdf = pd.DataFrame(cov_sdf, columns=labels, index=labels)
    file = os.path.join(resultspath, f'normalized_kidney_cov_sdf.csv')
    cov_sdf.to_csv(file)

    cov_lb = pd.DataFrame(cov_lb, columns=labels, index=labels)
    file = os.path.join(resultspath, f'normalized_kidney_cov_lb.csv')
    cov_lb.to_csv(file)




def build_spectral_feature_vectors(build_path):

    datapath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_shape_analysis')

    # Get baseline kidneys
    kidney_masks = db.series(datapath)
    kidney_masks = [k for k in kidney_masks if k[2][0] in ['Visit1', 'Baseline']]
    kidney_masks = kidney_masks[:4] # for debugging

    k0 = 32

    # Loop over cases
    for kidney in ['left', 'right']:
        spectral_features = []
        norm_masks = [k for k in kidney_masks if k[3][0] == f'normalized_{kidney}_kidney_mask']
        for kidney_mask in tqdm(enumerate(norm_masks), desc=f'Loop kidney {kidney}'):
            patient = kidney_mask[1]
            mask_norm = db.volume(kidney_mask).values
            coeffs = sdf.coeffs_from_mask(mask_norm)
            coeffs_dict = {patient: coeffs[:k0, :k0, :k0].flatten()}
            spectral_features.append(coeffs_dict)
        df_features = pd.DataFrame(spectral_features)
        df_file = os.path.join(resultspath, f'{kidney}_kidney_spectral_features.csv')
        df_features.to_csv(df_file)


def build_binary_feature_vectors(build_path):

    datapath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_shape_analysis')

    # Get baseline kidneys
    kidney_masks = db.series(datapath)
    kidney_masks = [k for k in kidney_masks if k[2][0] in ['Visit1', 'Baseline']]
    kidney_masks = kidney_masks[:4] # for debugging

    k0 = 32

    # Loop over cases
    for kidney in ['left', 'right']:
        features = []
        norm_masks = [k for k in kidney_masks if k[3][0] == f'normalized_{kidney}_kidney_mask']
        for kidney_mask in tqdm(enumerate(norm_masks), desc=f'Loop kidney {kidney}'):
            patient = kidney_mask[1]
            mask_norm = db.volume(kidney_mask).values
            coeffs_dict = {patient: mask_norm.flatten()}
            features.append(coeffs_dict)
        df_features = pd.DataFrame(features)
        df_file = os.path.join(resultspath, f'{kidney}_kidney_binary_features.csv')
        df_features.to_csv(df_file)


def principal_component_analysis(build_path):

    n_components = 10
    n_clusters = 3
    
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_shape_analysis')

    for kidney in ['left', 'right']:
        df_file = os.path.join(resultspath, f'{kidney}_kidney_features.csv')
        df_features = pd.read_csv(df_file)

        # Run PCA on feature matrix
        pca = PCA(n_components=n_components)
        features_reduced = pca.fit_transform(df_features.values)

        # Cluster shapes in PCA space
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(features_reduced)

        # Reconstruct shapes for the different features
        # coeffs = features_reduced[0].reshape((32,32,32)) 
        # mask_recon = sdf.mask_from_coeffs(coeffs)

