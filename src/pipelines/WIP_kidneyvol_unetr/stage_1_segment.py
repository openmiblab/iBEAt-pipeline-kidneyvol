import os
import logging

import numpy as np
import dbdicom as db
import miblab
import vreg

import utils.data


EXCLUDE = []


datapath = os.path.join(os.getcwd(), 'build', 'dixon_2_data') 
maskpath = os.path.join(os.getcwd(), 'build', 'kidney_unetr_1_segment') 
os.makedirs(maskpath, exist_ok=True)


# Set up logging
logging.basicConfig(
    filename=os.path.join(maskpath, 'error.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def segment_site(site, batch_size=None):

    sitedatapath = os.path.join(datapath, site, "Patients") 
    sitemaskpath = os.path.join(maskpath, site, "Patients")
    os.makedirs(sitemaskpath, exist_ok=True)

    # List of selected dixon series
    record = utils.data.dixon_record()

    # Get out phase series
    series = db.series(sitedatapath)
    series_out_phase = [s for s in series if s[3][0][-9:]=='out_phase']

    # Loop over the out-phase series
    count = 0
    for series_op in series_out_phase:

        # Patient and output study
        patient = series_op[1]
        study = series_op[2][0]
        series_op_desc = series_op[3][0]
        sequence = series_op_desc[:-10]

        # Skip those marked for exclusion
        if patient in EXCLUDE:
            continue

        # Skip if it is not the right sequence
        selected_sequence = utils.data.dixon_series_desc(record, patient, study)
        if sequence != selected_sequence:
            continue

        # Skip if the kidney masks already exist
        mask_study = [sitemaskpath, patient, (study, 0)]
        mask_series = mask_study + [(f'kidney_masks', 0)]
        if mask_series in db.series(mask_study):
            continue

        # Other source data series
        series_ip = series_op[:3] + [(sequence + '_in_phase', 0)]
        series_wi = series_op[:3] + [(sequence + '_water', 0)]
        series_fi = series_op[:3] + [(sequence + '_fat', 0)]

        # Select model to use
        if series_wi not in db.series(series_op[:3]):
           continue

        # Read the in- and out of phase volumes
        try:
            op = db.volume(series_op)
            ip = db.volume(series_ip)
            wi = db.volume(series_wi)
            fi = db.volume(series_fi)
        except Exception as e:
            logging.error(f"Patient {patient} - error reading I-O {sequence}: {e}")
            continue

        # Predict kidney masks
        try:
            array = np.stack((op.values, ip.values, wi.values, fi.values), axis=-1)
        except Exception as e:
            logging.error(f"{patient} {sequence} error building 4-channel input array: {e}")
            continue
        # vol = vreg.volume(array, op.affine)
        # rois = miblab.kidney_pc_dixon_unetr(vol, verbose=True)
        try:
            vol = vreg.volume(array, op.affine)
            rois = miblab.kidney_pc_dixon_unetr(vol, verbose=True)
        except Exception as e:
            logging.error(f"Error processing {patient} {sequence} with unetr: {e}")
            continue
            
        # Write in dicom as integer label arrays to save space
        db.write_volume(rois, mask_series, ref=series_op)
        
        count += 1 
        if batch_size is not None:
            if count >= batch_size:
                return
            


def all(batch_size=None):
    segment_site('Sheffield', batch_size)
    segment_site('Leeds', batch_size)
    segment_site('Bari', batch_size)


if __name__=='__main__':
    segment_site('Sheffield')
    # segment_site('Leeds')
    # segment_site('Bari')
    
    
    
    