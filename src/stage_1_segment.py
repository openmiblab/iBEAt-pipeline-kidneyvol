import os
import logging

import numpy as np
import dbdicom as db
import miblab
import torch


import utils.data
from utils import radiomics
from utils.constants import SITE_IDS


# These need fully manual segmentation
EXCLUDE = [ 
    '4128_055', # miblab nnunet No segmentation: large left kidney and tiny right kidney
    '7128_149', # miblab nnunet Segmentation failed: horseshoe kidney
]

# Exceptions: failed with nnunet for no obvious reason
TOTSEG = [
    ('7128_085', 'Baseline'),

    # Leeds
    ('4128_007', 'Baseline'),
    ('4128_010', 'Baseline'), # poor images
    ('4128_012', 'Baseline'),
    ('4128_013', 'Baseline'), # poor images
    ('4128_014', 'Baseline'),
    ('4128_015', 'Baseline'),
    ('4128_016', 'Baseline'),
    ('4128_017', 'Baseline'),
    ('4128_024', 'Baseline'),
    ('4128_030', 'Baseline'),
    ('4128_043', 'Baseline'),
    ('4128_051', 'Baseline'),
    ('4128_052', 'Baseline'),
    ('4128_053', 'Baseline'),
    ('4128_054', 'Baseline'),
    ('4128_061', 'Baseline'),

    # Sheffield
    ('7128_021', 'Baseline'),
    ('7128_026', 'Baseline'),
    ('7128_027', 'Baseline'),
    ('7128_033', 'Baseline'),
    ('7128_037', 'Baseline'),
    ('7128_038', 'Baseline'),
    ('7128_040', 'Baseline'),
    ('7128_044', 'Baseline'),
    ('7128_047', 'Baseline'),
    ('7128_055', 'Baseline'), # failed with nnunet
    ('7128_056', 'Baseline'),
    ('7128_059', 'Baseline'),
    ('7128_064', 'Baseline'),
    ('7128_067', 'Baseline'),
    ('7128_069', 'Baseline'),
    ('7128_072', 'Baseline'),
    ('7128_073', 'Baseline'),
    ('7128_074', 'Baseline'),
    ('7128_075', 'Baseline'),
    ('7128_076', 'Baseline'),
    ('7128_077', 'Baseline'),
    ('7128_080', 'Baseline'),
    ('7128_081', 'Baseline'),
    ('7128_082', 'Baseline'),
    ('7128_083', 'Baseline'),
    ('7128_084', 'Baseline'),
    ('7128_086', 'Baseline'),
    ('7128_087', 'Baseline'),
    ('7128_091', 'Baseline'),
    ('7128_092', 'Baseline'),
    ('7128_093', 'Baseline'),
    ('7128_094', 'Baseline'),
    ('7128_096', 'Baseline'),
    ('7128_101', 'Baseline'),
    ('7128_102', 'Baseline'),
    ('7128_104', 'Baseline'),
    ('7128_106', 'Baseline'),
    ('7128_109', 'Baseline'),
    ('7128_110', 'Baseline'),
    ('7128_111', 'Baseline'),
    ('7128_112', 'Baseline'),
    ('7128_113', 'Baseline'),
    ('7128_114', 'Baseline'), # very poor images
    ('7128_115', 'Baseline'),
    ('7128_116', 'Baseline'),
    ('7128_117', 'Baseline'),
    ('7128_118', 'Baseline'),
    ('7128_129', 'Baseline'),
    ('7128_132', 'Baseline'),
    ('7128_137', 'Baseline'),
    ('7128_140', 'Baseline'),
    ('7128_144', 'Baseline'),
    ('7128_146', 'Baseline'),
    ('7128_147', 'Baseline'),
    ('7128_148', 'Baseline'),
    ('7128_155', 'Baseline'),
    ('7128_156', 'Baseline'),
    ('7128_157', 'Baseline'),
    ('7128_160', 'Baseline'),
    ('7128_163', 'Baseline'),
    ('7128_164', 'Baseline'),
    ('7128_165', 'Baseline'),
    ('7128_166', 'Baseline'),

    ('2128_007', 'Baseline'),
    ('2128_009', 'Baseline'),
    ('2128_020', 'Baseline'),
    ('2128_028', 'Baseline'),
    ('2128_032', 'Baseline'),
    ('2128_040', 'Baseline'),
    ('2128_045', 'Baseline'),
    ('6128_001', 'Baseline'),
    ('6128_001', 'Followup'),
    ('6128_009', 'Baseline'),

    ('3128_007', 'Followup'),
    ('3128_014', 'Baseline'),
    ('3128_014', 'Followup'),
    ('3128_018', 'Baseline'),
    ('3128_019', 'Baseline'),
    ('3128_019', 'Followup'),
    ('3128_023', 'Baseline'),
    ('3128_024', 'Baseline'),
    ('3128_026', 'Baseline'),
    ('3128_026', 'Followup'),
    ('3128_031', 'Baseline'),
    ('3128_033', 'Followup'),
    ('3128_043', 'Baseline'),
    ('3128_044', 'Baseline'),
    ('3128_045', 'Baseline'),
    ('3128_047', 'Baseline'),
    ('3128_050', 'Baseline'),
    ('3128_056', 'Baseline'),
    ('3128_058', 'Baseline'),
    ('3128_059', 'Baseline'),
    ('3128_067', 'Baseline'),
    ('3128_067', 'Followup'),
    ('3128_070', 'Baseline'),
    ('3128_074', 'Baseline'),
    ('3128_074', 'Followup'),
    ('3128_078', 'Followup'),
    ('3128_080', 'Baseline'),
    ('3128_081', 'Baseline'),
    ('3128_086', 'Baseline'),
    ('3128_086', 'Followup'),
    ('3128_090', 'Baseline'),
    ('3128_091', 'Baseline'),
    ('3128_091', 'Followup'),
    ('3128_095', 'Baseline'),
    ('3128_095', 'Followup'),
    ('3128_096', 'Baseline'),
    ('3128_096', 'Followup'),
    ('3128_097', 'Baseline'),
    ('3128_098', 'Baseline'),
    ('3128_107', 'Baseline'),
    ('3128_114', 'Baseline'),
    ('3128_115', 'Baseline'),
    ('3128_116', 'Baseline'),
    ('3128_131', 'Baseline'),
    ('3128_133', 'Baseline'),
    ('3128_137', 'Baseline'),
]



def segment_site(build_path, group, site=None, batch_size=None):

    datapath = os.path.join(build_path, 'dixon', 'stage_2_data') 
    maskpath = os.path.join(build_path, 'kidneyvol', 'stage_1_segment') 
    os.makedirs(maskpath, exist_ok=True)

    if group == 'Controls':
        sitedatapath = os.path.join(datapath, group) 
        sitemaskpath = os.path.join(maskpath, group)
    else:
        sitedatapath = os.path.join(datapath, group, site) 
        sitemaskpath = os.path.join(maskpath, group, site)
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

        # Skip if the patient is not in the right site
        if patient[:4] not in SITE_IDS[site]:
            continue

        # Skip if it is not the right sequence
        selected_sequence = utils.data.dixon_series_desc(record, patient, study)
        if sequence != selected_sequence:
            continue

        # Skip if the kidney masks already exist
        mask_study = [sitemaskpath, patient, (study,0)]
        mask_series = mask_study + [(f'kidney_masks', 0)]
        if mask_series in db.series(mask_study):
            continue

        # Other source data series
        series_ip = series_op[:3] + [(sequence + '_in_phase', 0)]
        series_wi = series_op[:3] + [(sequence + '_water', 0)]
        series_fi = series_op[:3] + [(sequence + '_fat', 0)]

        # Read the in- and out of phase volumes
        try:
            op = db.volume(series_op)
            ip = db.volume(series_ip)
        except Exception as e:
            logging.error(f"Patient {patient} - error reading I-O {sequence}: {e}")
            continue

        # Select model to use
        if series_wi not in db.series(series_op[:3]):
            # If there are only 2 channels, use total segmentator
            model = 'totseg'
        elif (patient, study) in TOTSEG: 
            # Exception: failed with nnunet/unetr for no obvious reason
            model = 'totseg'
        else:
            # Default for 4-channel data is nnunet
            model = 'nnunet'

        # If autosegmentation does not work, draw rectangles in the middle slice 
        if patient in EXCLUDE:
            label_array = np.zeros(op.shape, dtype=np.int16)
            xm = int(np.round(op.shape[0]/2))
            ym = int(np.round(op.shape[1]/2))
            zm = int(np.round(op.shape[2]/2))
            label_array[xm+1:xm+10, ym+1:ym+10, zm] = 1
            label_array[xm-10:xm-1, ym-10:ym-1, zm] = 2

        # If there are only 2 channels, use total segmentator
        elif model=='totseg':
            try:
                device = 'gpu' if torch.cuda.is_available() else 'cpu'
                label_vol = miblab.totseg(op, cutoff=0.01, task='total_mr', device=device)
                # Extract kidneys only
                label_array = label_vol.values
                label_array[~np.isin(label_array, [2,3])] = 0
                # Relabel left and right
                label_array[label_array==3] = 1
                # Remove smaller disconnected clusters
                label_array = radiomics.largest_cluster_label(label_array)
            except Exception as e:
                logging.error(f"Error processing {patient} {sequence} with total segmentator: {e}")
                continue

        # If there are 4 channels, use miblab nnunet or unetr:
        else:

            # Read fat and water data
            try:
                wi = db.volume(series_wi)
                fi = db.volume(series_fi)
            except Exception as e:
                logging.error(f"Patient {patient} - error reading F-W {sequence}: {e}")
                continue

            # Predict kidney masks
            try:
                array = np.stack((op.values, ip.values, wi.values, fi.values), axis=-1)
            except Exception as e:
                logging.error(f"{patient} {sequence} error building 4-channel input array: {e}")
                continue
            try:
                label_array = miblab.kidney_pc_dixon(array, verbose=True)
            except Exception as e:
                logging.error(f"Error processing {patient} {sequence} with nnunet: {e}")
                continue
        
        db.write_volume((label_array, op.affine), mask_series, ref=series_op)

        count += 1 
        if batch_size is not None:
            if count >= batch_size:
                return

