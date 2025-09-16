import os
import logging
import shutil

from tqdm import tqdm
import numpy as np
import dbdicom as db
import vreg
import pydmr

from utils import data, radiomics



def concat(measurepath, group, site=None):
    if group == 'Controls':
        sitemeasurepath = os.path.join(measurepath, "Controls") 
    else:   
        sitemeasurepath = os.path.join(measurepath, "Patients", site)
    # Concatenate all dmr files of each subject
    patients = [f.path for f in os.scandir(sitemeasurepath) if f.is_dir()]
    for patient in patients:
        dir = os.path.join(sitemeasurepath, patient)
        dmr_files = [f for f in os.listdir(dir) if f.endswith('.dmr.zip')]
        dmr_files = [os.path.join(dir, f) for f in dmr_files]
        dmr_file = os.path.join(sitemeasurepath, f'{patient}_results')
        pydmr.concat(dmr_files, dmr_file)
        shutil.rmtree(dir)


def measure(build_path, group, site=None):

    datapath = os.path.join(build_path, 'dixon', 'stage_2_data')
    automaskpath = os.path.join(build_path, 'kidneyvol', 'stage_1_segment')
    editmaskpath = os.path.join(build_path, 'kidneyvol', 'stage_3_edit')
    measurepath = os.path.join(build_path, 'kidneyvol', 'stage_5_measure')
    os.makedirs(measurepath, exist_ok=True)

    if group == 'Controls':
        sitedatapath = os.path.join(datapath, "Controls") 
        siteautomaskpath = os.path.join(automaskpath, "Controls")
        siteeditmaskpath = os.path.join(editmaskpath, "Controls")
        sitemeasurepath = os.path.join(measurepath, "Controls")         
    else:
        sitedatapath = os.path.join(datapath, "Patients", site) 
        siteautomaskpath = os.path.join(automaskpath, "Patients", site)
        siteeditmaskpath = os.path.join(editmaskpath, "Patients", site)
        sitemeasurepath = os.path.join(measurepath, "Patients", site)

    record = data.dixon_record()
    class_map = {1: "kidney_left", 2: "kidney_right"}
    all_editmasks = db.series(siteeditmaskpath)
    all_automasks = db.series(siteautomaskpath)

    for automask in tqdm(all_automasks, desc='Extracting metrics'):

        patient, study, series = automask[1], automask[2][0], automask[3][0]
        dir = os.path.join(sitemeasurepath, patient)
        os.makedirs(dir, exist_ok=True)

        # If the patient results exist, skip
        dmr_file = os.path.join(sitemeasurepath, f'{patient}_results')
        if os.path.exists(f'{dmr_file}.dmr.zip'):
            continue

        sequence = data.dixon_series_desc(record, patient, study)
        data_study = [sitedatapath, patient, (study, 0)]
        all_data_series = db.series(data_study)

        # Get mask volume (edited if it exists, else automated)
        editmask = [siteeditmaskpath, patient, (study, 0), ('kidney_masks', 0)]
        if editmask in all_editmasks:
            mask_series = editmask
        else:
            mask_series = automask
        vol = db.volume(mask_series)
        
        # Loop over the classes
        for idx, roi in class_map.items():

            # Skip if file exists
            dmr_file = os.path.join(dir, f"{study}_{series}_{roi}")
            if os.path.exists(f'{dmr_file}.dmr.zip'):
                continue

            # Binary mask
            mask = (vol.values==idx).astype(np.float32)
            if np.sum(mask) == 0:
                continue

            roi_vol = vreg.volume(mask, vol.affine)
            dmr = {'data':{}, 'pars':{}}

            # Get skimage features
            try:
                results = radiomics.volume_features(roi_vol, roi)
            except Exception as e:
                logging.error(f"Patient {patient} {roi} - error computing ski-shapes: {e}")
            else:
                dmr['data'] = dmr['data'] | {p: v[1:] for p, v in results.items()}
                dmr['pars'] = dmr['pars'] | {(patient, study, p): v[0] for p, v in results.items()}

            # Get radiomics shape features
            try:
                results = radiomics.shape_features(roi_vol, roi)
            except Exception as e:
                logging.error(f"Patient {patient} {roi} - error computing radiomics-shapes: {e}")
            else:
                dmr['data'] = dmr['data'] | {p:v[1:] for p, v in results.items()}
                dmr['pars'] = dmr['pars'] | {(patient, study, p): v[0] for p, v in results.items()}

            # Get radiomics texture features
            if roi in ['kidney_left', 'kidney_right']: # computational issues with larger ROIs.
                for img in ['out_phase', 'in_phase', 'fat', 'water']:
                    img_series = [sitedatapath, patient, (study, 0), (f"{sequence}_{img}", 0)]
                    if img_series not in all_data_series:
                        continue # Need a different solution here - compute assuming water dominant
                    img_vol = db.volume(img_series)
                    try:
                        results = radiomics.texture_features(roi_vol, img_vol, roi, img)
                    except Exception as e:
                        logging.error(f"Patient {patient} {roi} {img} - error computing radiomics-texture: {e}")
                    else:
                        dmr['data'] = dmr['data'] | {p:v[1:] for p, v in results.items()}
                        dmr['pars'] = dmr['pars'] | {(patient, study, p): v[0] for p, v in results.items()}

            # Write results to file
            pydmr.write(dmr_file, dmr)

        # Both kidneys texture
        roi = 'kidneys_both'
        dmr_file = os.path.join(dir, f"{study}_{series}_{roi}.dmr.zip")
        if os.path.exists(dmr_file):
            continue
        class_index = {roi:idx for idx,roi in class_map.items()}
        vol = db.volume(mask_series)
        lk_mask = (vol.values==class_index['kidney_left']).astype(np.float32)
        rk_mask = (vol.values==class_index['kidney_right']).astype(np.float32)
        mask = lk_mask + rk_mask
        if np.sum(mask) == 0:
            continue
        roi_vol = vreg.volume(mask, vol.affine)
        dmr = {'data':{}, 'pars':{}}

        # Get radiomics texture features
        for img in ['out_phase', 'in_phase', 'fat', 'water']:
            img_series = [sitedatapath, patient, (study, 0), (f"{sequence}_{img}", 0)]
            if img_series not in all_data_series:
                continue
            img_vol = db.volume(img_series)
            try:
                results = radiomics.texture_features(roi_vol, img_vol, roi, img)
            except Exception as e:
                logging.error(f"Patient {patient} {roi} {img} - error computing radiomics-texture: {e}")
            else:
                dmr['data'] = dmr['data'] | {p:v[1:] for p, v in results.items()}
                dmr['pars'] = dmr['pars'] | {(patient, study, p): v[0] for p, v in results.items()}

        # Write results to file
        pydmr.write(dmr_file, dmr)

    concat(measurepath, site)
