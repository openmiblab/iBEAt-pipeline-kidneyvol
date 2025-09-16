import os
import logging

import numpy as np
import dbdicom as db
import miblab
import torch

import utils.data


datapath = os.path.join(os.getcwd(), 'build', 'dixon_2_data') 
maskpath = os.path.join(os.getcwd(), 'build', 'vat_1_segment') 
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

        # Skip if it is not the right sequence
        selected_sequence = utils.data.dixon_series_desc(record, patient, study)
        if sequence != selected_sequence:
            continue

        # Skip if the masks already exist
        mask_study = [sitemaskpath, patient, (study,0)]
        mask_series = mask_study + [(f'tissue_types_mr', 0)]
        if mask_series in db.series(mask_study):
            continue

        # Other source data series
        series_wi = series_op[:3] + [(sequence + '_water', 0)]

        # Read volumes
        if series_wi in db.series(series_op[:3]):
            vol = db.volume(series_wi)
        else:
            vol = db.volume(series_op)

        # Perform the segmentation
        try:
            device = 'gpu' if torch.cuda.is_available() else 'cpu'
            label_vol = miblab.totseg(vol, cutoff=0.01, task='tissue_types_mr', device=device)
        except Exception as e:
            logging.error(f"Error processing {patient} {sequence} with total segmentator: {e}")
            continue

        # Save results
        db.write_volume(label_vol, mask_series, ref=series_op)

        count += 1 
        if batch_size is not None:
            if count >= batch_size:
                return



def all():
    segment_site('Sheffield')
    segment_site('Leeds')
    segment_site('Bari')


if __name__=='__main__':
    segment_site('Sheffield')
    segment_site('Leeds')
    segment_site('Bari')
    
    
    