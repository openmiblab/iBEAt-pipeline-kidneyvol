import os

import dbdicom as db
import vreg
import numpy as np
from totalsegmentator.map_to_binary import class_map
import napari
from scipy.fftpack import dctn, idctn

from utils import normalize, render, lb, sdf

DIXONS = os.path.join(os.getcwd(), 'build', 'dixon', 'stage_2_data')
SEGMENTATIONS = os.path.join(os.getcwd(), 'build', 'kidneyvol', 'stage_3_edit')

patient = '1128_003'
study = 'Visit1'
study = 'Baseline'
task = 'kidney_masks'


def display_surface_lb():

    series = [SEGMENTATIONS, patient, study, task]
    vol = db.volume(series)
    mask = vol.values == 2
    mesh = lb.mask_to_mesh(mask) 
    coeffs, eigvals, recon_mesh = lb.process(mesh, k=100, threshold=15)
    for i, c in enumerate(coeffs):
        print(i, eigvals[i], float(np.linalg.norm(c)))
    render.visualize_surface_reconstruction(mesh, recon_mesh, opacity=(0.3,0.3))


def display_surface_sdf():
    series = [SEGMENTATIONS, patient, study, task]
    vol = db.volume(series)
    
    # Normalize and display
    mask = vol.values == 2
    mask_norm, params = normalize.normalize_mask(mask)
    render.display_volumes(mask, mask_norm)

    # Visualize
    size = 20
    coeffs_trunc, mask_norm_recon = sdf.compress(mask_norm, 3 * [size])

    # Visualize
    render.display_volumes(mask_norm, mask_norm_recon)


def display_normalized():
    series = [SEGMENTATIONS, patient, study, task]
    vol = db.volume(series)
    voxel_size = vol.spacing
    spacing_norm = 1.0
    volume_norm = 1e6
    voxel_size_norm = 3 * [spacing_norm]

    rk_mask = vol.values == 2
    rk_mask_norm, _ = normalize.volume_normalize_mask(rk_mask, voxel_size, spacing_norm, volume_norm, mirror_axis=None)
    render.display_kidney_normalization(rk_mask, rk_mask_norm, voxel_size, voxel_size_norm, title='Right kidney')

    lk_mask = vol.values == 1
    lk_mask_norm, _ = normalize.volume_normalize_mask(lk_mask, voxel_size, spacing_norm, volume_norm, mirror_axis=0)
    render.display_kidney_normalization(np.flip(lk_mask, 0), lk_mask_norm, voxel_size, voxel_size_norm, title='Left kidney flipped')

    return
    # render.display_normalized_kidneys(rk_mask_norm, lk_mask_norm, voxel_size_norm)

    # Check compression

    cutoff = 64

    rk_mask_recon, _ = sdf.compress(rk_mask_norm, 3 * [cutoff])
    render.compare_processed_kidneys(rk_mask_norm, rk_mask_recon, voxel_size_norm)

    lk_mask_recon, _ = sdf.compress(lk_mask_norm, 3 * [cutoff])
    render.compare_processed_kidneys(lk_mask_norm, lk_mask_recon, voxel_size_norm)

    print('\nRight kidney volume')
    print(f"Target: {volume_norm / 1e6} Litre")
    print(f"Actual: {rk_mask_norm.sum() * spacing_norm ** 3 / 1e6} Litre")
    print(f"Compressed: {rk_mask_recon.sum() * spacing_norm ** 3 / 1e6} Litre")
    print(f"Loss: {np.around(100 * np.abs(rk_mask_recon.sum() - rk_mask_norm.sum()) / rk_mask_norm.sum(), 2) } %")
    print('\nLeft kidney volume')
    print(f"Target: {volume_norm / 1e6} Litre")
    print(f"Actual: {lk_mask_norm.sum() * spacing_norm ** 3 / 1e6} Litre")
    print(f"Compressed: {lk_mask_recon.sum() * spacing_norm ** 3 / 1e6} Litre")
    print(f"Loss: {np.around(100 * np.abs(lk_mask_recon.sum() - lk_mask_norm.sum()) / lk_mask_norm.sum(), 2) } %")



if __name__=='__main__':
    # display_surface_lb()
    # display_surface_sdf()
    display_normalized()
