import os
import logging
import json
import argparse

from tqdm import tqdm
import numpy as np
import dbdicom as db
import pyvista as pv


from utils import sdf

# Configure logging once, at the start of your script
logging.basicConfig(
    filename='parametrize.log',      # log file name
    level=logging.INFO,           # log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # log format
)






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





if __name__ == '__main__':

    DATA_DIR = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\kidneyvol\stage_7_normalized"
    RESULTS_DIR = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\kidneyvol\stage_7_normalized_npz"

