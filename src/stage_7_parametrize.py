import os
import logging

from tqdm import tqdm
import numpy as np
import dbdicom as db
import pyvista as pv
import pandas as pd
import vreg
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from utils import normalize, sdf


def normalize_kidneys(build_path):

    datapath = os.path.join(build_path, 'kidneyvol', 'stage_3_edit')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')

    kidney_labels = db.series(datapath)

    spacing_norm = 1.0
    volume_norm = 1e6

    vol = None
    for kidney_label in tqdm(kidney_labels, desc=f'Normalizing kidneys..'):
        study = [resultspath] + kidney_label[1:3]
        # Right kidney
        rk_series = study + [('normalized_right_kidney_mask', 0)]
        if rk_series not in db.series(study):
            try:
                vol = db.volume(kidney_label)
                rk_mask = vol.values == 2
                rk_mask_norm, _ = normalize.volume_normalize_mask(rk_mask, vol.spacing, spacing_norm, volume_norm, mirror_axis=None)
                rk_vol_norm = vreg.volume(rk_mask_norm.astype(int))
                db.write_volume(rk_vol_norm, rk_series, ref=kidney_label)
            except Exception as e:
                logging.error(f"Error normalizing right kidney of patient {rk_series[1]} in study {rk_series[2][0]}: {e}")

        # Left kidney
        lk_series = study + [('normalized_left_kidney_mask', 0)]
        if lk_series not in db.series(study):
            try:
                if vol is None:
                    vol = db.volume(kidney_label)
                lk_mask = vol.values == 1
                lk_mask_norm, _ = normalize.volume_normalize_mask(lk_mask, vol.spacing, spacing_norm, volume_norm, mirror_axis=0)
                lk_vol_norm = vreg.volume(lk_mask_norm.astype(int))
                db.write_volume(lk_vol_norm, lk_series, ref=kidney_label)
            except Exception as e:
                logging.error(f"Error normalizing left kidney of patient {lk_series[1]} in study {lk_series[2][0]}: {e}")
        
        vol = None

def display_all_normalizations(build_path, group=None, site=None):

    # Paths
    datapath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_shape_analysis')
    os.makedirs(resultspath, exist_ok=True)
    if group is None:
        prefix = ''
    elif group == 'Controls':
        datapath = os.path.join(datapath, "Controls")
        prefix = 'controls_'
    else:
        datapath = os.path.join(datapath, "Patients", site) 
        prefix = f"patients_{site}_"

    # Plot settings
    aspect_ratio = 16/9
    width = 150
    height = 150

    # Get baseline kidneys
    kidney_masks = db.series(datapath)
    kidney_masks = [k for k in kidney_masks if k[2][0] in ['Visit1', 'Baseline']]

    for kidney in ['right', 'left']:

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



def build_dice_correlations(build_path):

    datapath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_7_shape_analysis')

    # Get baseline kidneys
    kidney_masks = db.series(datapath)
    kidney_masks = [k for k in kidney_masks if k[2][0] in ['Visit1', 'Baseline']]
    kidney_masks = kidney_masks[:4] # for debugging
    n_cases = len(kidney_masks)
    
    for kidney in ['left', 'right']:
        dice = np.zeros((n_cases, n_cases))
        norm_masks = [k for k in kidney_masks if k[3][0] == f'normalized_{kidney}_kidney_mask']
        for i, kidney_mask_i in tqdm(enumerate(norm_masks[:-1]), desc=f'Outer loop kidney {kidney}'):
            mask_norm_i = db.volume(kidney_mask_i).values
            for j, kidney_mask_j in tqdm(enumerate(norm_masks[i+1:]), desc=f'Inner loop kidney {kidney}'):
                mask_norm_j = db.volume(kidney_mask_j).values
                dice[i,j] = normalize.dice_coefficient(mask_norm_i, mask_norm_j)
        dice = dice + dice.T
        for i in range(n_cases):
            dice[i,i] = 1
        patients = [k[1] for k in norm_masks]
        dice = pd.DataFrame(dice, columns=patients, index=patients)
        file = os.path.join(resultspath, f'{kidney}_kidney_dice.csv')
        dice.to_csv(file)


def cluster_dice_correlations(build_path):
    pass

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

if __name__=='__main__':
    pass