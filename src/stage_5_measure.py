import os
import logging
import shutil
from pathlib import Path
from datetime import date

from tqdm import tqdm
import numpy as np
import dbdicom as db
import vreg
import pydmr
import pandas as pd

from utils import data, radiomics


def combine(build_path):
    """
    Concatenate all dmri files in a folder into a single dmr file. 
    Create long and wide format csv's for export.
    """
    measurepath = os.path.join(build_path, 'kidneyvol', 'stage_5_measure')
    for group in ['Controls', 'Patients']:

        # Combine all dmr files into one
        folder = os.path.join(measurepath, group) 
        folder = Path(folder)
        dmr_files = list(folder.rglob("*.dmr.zip"))  
        if dmr_files == []:
            continue
        dmr_files = [str(f) for f in dmr_files]
        dmr_file = os.path.join(measurepath, f'{group}_kidneyvol')
        pydmr.concat(dmr_files, dmr_file)


def export_to_redcap(build_path):
    """
    Create export for upload to redcap repository
    """
    today = date.today().strftime("%Y-%m-%d")
    measurepath = os.path.join(build_path, 'kidneyvol', 'stage_5_measure')
    resultspath = os.path.join(build_path, 'kidneyvol', 'stage_5_measure', 'RedCap')
    os.makedirs(resultspath, exist_ok=True)

    def visit_nr(value):
        if value == 'Baseline':
            return 0
        if value == 'Followup':
            return 2
        if value[:5] == 'Visit':
            return int(value[5]) - 1
        
    def fix_exeter_volunteer(harmonized_id, visit_nr):
        # Correct a mistake in ID 
        # Exeter volunteer 3 is the same person as volunteer 1
        # This needs to be removed when the issue is fixed at the source
        if harmonized_id == 'iBE-3128C03':
            harmonized_id = 'iBE-3128C01'
            visit_nr += 2
        return harmonized_id, visit_nr

    for group in ['Controls', 'Patients']:
        dmr_file = os.path.join(measurepath, f'{group}_kidneyvol')
        dmr = pydmr.read(dmr_file)

        # Append parsed biomarkers in the dictionary
        dmr_output_file = os.path.join(resultspath, f'{group}_KidneyShape_{today}')
        dmr['columns'] = ['body_part', 'image', 'biomarker_category', 'biomarker']
        for p in dmr['data']:
            parts = p.split('-')
            # For intrinsic markers add image 'mask'
            if len(parts) == 3:
                parts = [parts[0]] + ['mask'] + parts[1:]
            dmr['data'][p] += parts

        # Change PatientIDs to central format
        pars_harmonized = {}
        for p,v in dmr['pars'].items():
            harmonized_id = f"iBE-{p[0].replace('_','')}"
            visit = visit_nr(p[1])
            harmonized_id, visit = fix_exeter_volunteer(harmonized_id, visit)
            pars_harmonized[(harmonized_id, visit, p[2])] = v
        dmr['pars'] = pars_harmonized
        pydmr.write(dmr_output_file, dmr)

        # 1. Save in long format with additional columns
        long_format_file = os.path.join(resultspath, f'{group}_KidneyShape_{today}.csv')
        pydmr.pars_to_long(dmr_output_file, long_format_file)
        # Replace column names

        # 2. Save in wide format
        wide_format_file = os.path.join(resultspath, f'{group}_KidneyShape_{today}_wide.csv')
        pydmr.pars_to_wide(dmr_output_file, wide_format_file)

        # Replace column names
        new_cols = {
            "subject": "harmonized_id",
            "study": "visit_nr",
            "value": "result",
        }
        df = pd.read_csv(long_format_file)
        df.rename(columns=new_cols, inplace=True)
        df.to_csv(long_format_file, index=False)
        df = pd.read_csv(wide_format_file)
        df.rename(columns=new_cols, inplace=True)
        df.to_csv(wide_format_file, index=False)
        

def measure_shape(build_path, group, site=None):

    editmaskpath = os.path.join(build_path, 'kidneyvol', 'stage_3_edit')
    measurepath = os.path.join(build_path, 'kidneyvol', 'stage_5_measure')
    if group == 'Controls':
        maskpath = os.path.join(editmaskpath, "Controls")
        measurepath = os.path.join(measurepath, "Controls")         
    else:
        maskpath = os.path.join(editmaskpath, "Patients", site)
        measurepath = os.path.join(measurepath, "Patients", site)

    class_map = {1: "kidney_left", 2: "kidney_right"}
    all_masks = db.series(maskpath)

    for mask_series in tqdm(all_masks, desc='Extracting metrics'):

        patient, study, series = mask_series[1], mask_series[2][0], mask_series[3][0]
        dir = os.path.join(measurepath, patient)
        os.makedirs(dir, exist_ok=True)

        # If the patient results exist, skip
        dmr_file = os.path.join(measurepath, f'{patient}_results')
        if os.path.exists(f'{dmr_file}.dmr.zip'):
            continue

        # Get mask volume (edited if it exists, else automated)
        vol = db.volume(mask_series, verbose=0)
        
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
            _measure_kidney_mask_shape(roi_vol, roi, dmr_file, mask_series)

    # Concatenate all dmr files of each subject into a single 
    # dmr file and delete the originals.
    if group == 'Controls':
        sitemeasurepath = os.path.join(measurepath, "Controls") 
    else:   
        sitemeasurepath = os.path.join(measurepath, "Patients", site)

    patients = [f.path for f in os.scandir(sitemeasurepath) if f.is_dir()]
    for patient in patients:
        dir = os.path.join(sitemeasurepath, patient)
        dmr_files = [f for f in os.listdir(dir) if f.endswith('.dmr.zip')]
        dmr_files = [os.path.join(dir, f) for f in dmr_files]
        dmr_file = os.path.join(sitemeasurepath, f'{patient}_results')
        pydmr.concat(dmr_files, dmr_file)
        shutil.rmtree(dir)


def measure_normalized_mask_shape(build_path):

    maskpath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')
    measurepath = os.path.join(build_path, 'kidneyvol', 'stage_7_tmp')
    os.makedirs(measurepath, exist_ok=True)

    all_masks = db.series(maskpath)
    all_dmr_files = []

    for mask_series in tqdm(all_masks, desc='Extracting metrics'):

        patient, series = mask_series[1], mask_series[3][0]

        # If the results exist, skip
        roi = 'kidney_left' if 'left' in series else 'kidney_right'
        dmr_file = os.path.join(measurepath, f'{patient}_{roi}_results')
        if os.path.exists(f'{dmr_file}.dmr.zip'):
            continue

        # Get mask volume (edited if it exists, else automated)
        vol = db.volume(mask_series, verbose=0)

        # Binary mask
        mask = vol.values.astype(np.float32)
        if np.sum(mask) == 0:
            continue

        roi_vol = vreg.volume(mask, vol.affine)
        _measure_kidney_mask_shape(roi_vol, roi, dmr_file, mask_series)

        all_dmr_files.append(dmr_file)

    dmr_file = os.path.join(build_path, 'kidneyvol', f'stage_7_normalized_measures')
    pydmr.concat(all_dmr_files, dmr_file)
    shutil.rmtree(measurepath)


def _measure_kidney_mask_shape(roi_vol, roi, dmr_file, series):

    patient, study = series[1], series[2][0]

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

    # Write results to file
    pydmr.write(dmr_file, dmr)


def measure_texture(build_path, group, site=None):

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

    concat_patient(measurepath, site)
