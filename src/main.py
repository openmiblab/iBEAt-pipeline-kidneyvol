import os
import logging

import stage_0_restore
import stage_1_segment
import stage_2_display
import stage_3_edit
import stage_4_display
import stage_5_measure
import stage_6_archive


BUILD_PATH = os.path.join(os.getcwd(), 'build')
ARCHIVE_PATH = "G:\\Shared drives\\iBEAt_Build"


# Set up logging
logging.basicConfig(
    filename=os.path.join(BUILD_PATH, 'error.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_preprocessing():

    group = 'Controls'
    stage_0_restore.dixons(ARCHIVE_PATH, BUILD_PATH, group, site)
    stage_1_segment.segment_site(BUILD_PATH, group, site)
    stage_2_display.mosaic(BUILD_PATH, group, site)

    group = 'Patients'
    for site in ['Exeter', 'Leeds', 'Bari', 'Bordeaux', 'Sheffield', 'Turku']:
        stage_0_restore.dixons(ARCHIVE_PATH, BUILD_PATH, group, site)
        stage_1_segment.segment_site(BUILD_PATH, group, site)
        stage_2_display.mosaic(BUILD_PATH, group, site)
        stage_3_edit.auto_masks(BUILD_PATH, group, site)



def run_manual_section():

    group = 'Controls'
    stage_3_edit.auto_masks(BUILD_PATH, group, site)

    group = 'Patients'
    for site in ['Exeter', 'Leeds', 'Bari', 'Bordeaux', 'Sheffield', 'Turku']:
        stage_3_edit.auto_masks(BUILD_PATH, group, site)



def run_postprocessing():

    group = 'Controls'
    stage_4_display.mosaic(BUILD_PATH, group, site)
    stage_5_measure.measure(BUILD_PATH, group, site)
    stage_6_archive.autosegmentation(BUILD_PATH, ARCHIVE_PATH, group, site)
    stage_6_archive.edits(BUILD_PATH, ARCHIVE_PATH, group, site)

    group = 'Patients'
    for site in ['Exeter', 'Leeds', 'Bari', 'Bordeaux', 'Sheffield', 'Turku']:
        stage_4_display.mosaic(BUILD_PATH, group, site)
        stage_5_measure.measure(BUILD_PATH, group, site)
        stage_6_archive.autosegmentation(BUILD_PATH, ARCHIVE_PATH, group, site)
        stage_6_archive.edits(BUILD_PATH, ARCHIVE_PATH, group, site)



if __name__ == '__main__':
    run_preprocessing()
    run_manual_section()
    run_postprocessing()