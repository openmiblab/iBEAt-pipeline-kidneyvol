import os
import logging

import numpy as np
import skimage
from dipy.segment.mask import median_otsu
import scipy.ndimage as ndi
import dbdicom as db

import utils



def compute(build_path, group, site=None):
    # Get dixon op, fat image and edited kidney masks
    # compute rsf and save in dedicated database along with op and fat mask for editing
    datapath = os.path.join(build_path, 'dixon', 'stage_2_data') 
    kidneypath = os.path.join(build_path, 'kidneyvol', 'stage_3_edit')
    rsfpath = os.path.join(build_path, 'kidneyvol', 'stage_5_rsf') 
    os.makedirs(rsfpath, exist_ok=True)

    if group == 'Controls':
        sitedatapath = os.path.join(datapath, group) 
        sitekidneypath = os.path.join(kidneypath, group)
        sitersfpath = os.path.join(rsfpath, group)
    else:
        sitedatapath = os.path.join(datapath, group, site)
        sitekidneypath = os.path.join(kidneypath, group, site) 
        sitersfpath = os.path.join(rsfpath, group, site)
    os.makedirs(sitersfpath, exist_ok=True)

    # List of selected dixon series
    record = utils.data.dixon_record()

    # Get out phase series
    series = db.series(sitedatapath)
    series_out_phase = [s for s in series if s[3][0][-9:]=='out_phase']

    # Get all kidney masks
    all_kidney_masks = db.series(sitekidneypath)
    all_rsf_masks = db.series(sitersfpath)

    # Loop over the out-phase series
    for series_op in series_out_phase:

        # Patient and output study
        patient = series_op[1]
        study = series_op[2][0]
        series_op_desc = series_op[3][0]
        sequence = series_op_desc[:-10]

        # Skip if it is not the right sequence
        selected_sequence = utils.data.dixon_series_desc(record, patient, study)
        if sequence != selected_sequence:
            continue

        # Skip if fat image does not exist
        series_fi = series_op[:3] + [(sequence + '_fat', 0)]
        if series_fi not in series:
            continue

        # Skip if the kidney mask does not exist
        kidney_masks = [sitekidneypath, patient, (study, 0), ('kidney_masks', 0)]
        if kidney_masks not in all_kidney_masks:
            continue

        # Skip if the rsf masks already exist
        rsf_study = [sitersfpath, patient, (study,0)]
        rsf_series = rsf_study + [(f'rsf_masks', 0)]
        if rsf_series in all_rsf_masks:
            continue

        # Read the data
        try:
            fi = db.volume(series_fi)
            op = db.volume(series_op)
            kidney_label = db.volume(kidney_masks)
        except Exception as e:
            logging.error(f"Patient {patient} - error reading I-O {sequence}: {e}")
            continue

        try:
            rsf_label = renal_sinus_fat(fi.values, kidney_label.values)
        except Exception as e:
            logging.error(f"Error computing RSF for {patient} {sequence}: {e}")
            continue

        # Save results
        db.write_volume((rsf_label, op.affine), rsf_series, ref=series_op)

        # Include source images for QC and editing
        db.write_volume(fi, rsf_study + [f'dixon_fat'], ref=series_fi)
        db.write_volume(op, rsf_study + [f'dixon_out_phase'], ref=series_fi)



def renal_sinus_fat(fat, kidneys):

    #rk=2, lk=1

    rsf = np.zeros(kidneys.shape)
    fat_mask = median_otsu_2d(fat, median_radius=1, numpass=1)
    for kidney in [1, 2]:
        mask = (kidneys==kidney).astype(int)
        kidney_hull = convex_hull_image_3d(mask)
        sinus_fat = fat_mask * kidney_hull 
        #sinus_fat_open = skimage.opening_3d(sinus_fat)
        sinus_fat_largest = extract_largest_cluster_3d(sinus_fat)
        # closing after selecting largest cluster
        rsf[sinus_fat_largest] = kidney

    return rsf


def median_otsu_2d(array, **kwargs):
    mask = np.empty(array.shape)
    for z in range(array.shape[2]):
        image = np.squeeze(array[:,:,z])
        _, mask[:,:,z] = median_otsu(image, **kwargs)
    return mask


def convex_hull_image_3d(array, **kwargs):
    volume = np.around(array)
    return skimage.morphology.convex_hull_image(volume,  **kwargs)


def extract_largest_cluster_3d(array, **kwargs):

    label_img, cnt = ndi.label(array, **kwargs)
    # Find the label of the largest feature
    labels = range(1,cnt+1)
    size = [np.count_nonzero(label_img==l) for l in labels]
    max_label = labels[size.index(np.amax(size))]
    # Create a mask corresponding to the largest feature
    label_img = label_img==max_label
    #label_img = label_img[label_img==max_label]
    #label_img /= max_label
    return  label_img