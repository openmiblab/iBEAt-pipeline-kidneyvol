import os

import logging
from pipelines import totseg


LOCALPATH = os.path.join(os.getcwd(), 'build')
SHAREDPATH = os.path.join("G:\\Shared drives", "iBEAt_Build")


# Set up logging
logging.basicConfig(
    filename=os.path.join(LOCALPATH, 'error.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_totseg():

    # Define input and outputh paths here

    # group = 'Controls'
    # totseg.stage_0_restore.dixons(LOCALPATH, group)
    # totseg.stage_1_segment.segment(group)
    # totseg.stage_2_display.mosaic(group)
    # totseg.stage_2_display.mosaic(group, organs=['pancreas', 'liver'])
    # totseg.stage_4_archive.autosegmentation(LOCALPATH, SHAREDPATH, group)
    # totseg.stage_4_archive.displays(LOCALPATH, SHAREDPATH, group)

    group = 'Patients'
    sites = ['Bari', 'Bordeaux', 'Exeter', 'Leeds', 'Sheffield', 'Turku']
    for site in ['Sheffield']:
        totseg.stage_0_restore.dixons(LOCALPATH, SHAREDPATH, group, site)
        totseg.stage_1_segment.segment(LOCALPATH, group, site)
        totseg.stage_2_display.mosaic(LOCALPATH, group, site)
        totseg.stage_2_display.mosaic(LOCALPATH, group, site, organs=['pancreas', 'liver'])
        totseg.stage_4_archive.autosegmentation(LOCALPATH, SHAREDPATH, group, site)
        totseg.stage_4_archive.displays(LOCALPATH, SHAREDPATH, group, site)



if __name__ == '__main__':
    run_totseg()