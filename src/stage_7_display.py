import os
import logging
import json
import argparse

from tqdm import tqdm
import numpy as np
import dbdicom as db
import pyvista as pv


from utils import sdf, constants

# Configure logging once, at the start of your script
logging.basicConfig(
    filename='parametrize.log',      # log file name
    level=logging.INFO,           # log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # log format
)




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



if __name__ == '__main__':

    DATA_DIR = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\kidneyvol\stage_7_normalized"
    RESULTS_DIR = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\kidneyvol\stage_7_normalized_npz"

