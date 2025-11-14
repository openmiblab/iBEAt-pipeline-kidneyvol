import os
import logging

import stage_0_restore
import stage_1_segment
import stage_2_display
import stage_3_edit
import stage_4_display
import stage_5_measure
import stage_5_rsf
import stage_6_archive
import stage_7_parametrize
import stage_7_corr


BUILD_PATH = os.path.join(os.getcwd(), 'build')
DATA_PATH = os.path.join(os.getcwd(), 'src', 'data')
ARCHIVE_PATH = "G:\\Shared drives\\iBEAt_Build"
os.makedirs(BUILD_PATH, exist_ok=True)


# Set up logging
logging.basicConfig(
    filename=os.path.join(BUILD_PATH, 'error.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def build_canvas():

    # stage_0_restore.dixons(ARCHIVE_PATH, BUILD_PATH, 'Patients', 'Leeds')
    stage_1_segment.compute_canvas(BUILD_PATH, 'Patients', 'Leeds')
    stage_1_segment.display_canvas(BUILD_PATH, 'Patients', 'Leeds')


def preprocess_rsf():
    # stage_0_restore.edited_segmentations(ARCHIVE_PATH, BUILD_PATH, 'Patients', 'Leeds')
    stage_5_rsf.compute(BUILD_PATH, 'Patients', 'Leeds')


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
    # stage_4_display.mosaic(BUILD_PATH, group)
    # stage_5_measure.measure_shape(BUILD_PATH, group)
    # stage_6_archive.autosegmentation(BUILD_PATH, ARCHIVE_PATH, group)
    # stage_6_archive.edits(BUILD_PATH, ARCHIVE_PATH, group)

    # group = 'Patients'
    # for site in ['Exeter', 'Leeds', 'Bari', 'Bordeaux', 'Sheffield', 'Turku']:
        # stage_4_display.mosaic(BUILD_PATH, group, site)
        # stage_5_measure.measure_shape(BUILD_PATH, group, site)
        # stage_6_archive.autosegmentation(BUILD_PATH, ARCHIVE_PATH, group, site)
        # stage_6_archive.edits(BUILD_PATH, ARCHIVE_PATH, group, site)

    # stage_5_measure.measure_shape(BUILD_PATH, 'Patients', 'Bari')
    # stage_6_archive.edits(BUILD_PATH, ARCHIVE_PATH, 'Patients', 'Bari')

    # stage_5_measure.combine(BUILD_PATH)
    # stage_5_measure.export_to_redcap(BUILD_PATH)


def run_shape_analysis():

    # stage_0_restore.normalized_kidneys(ARCHIVE_PATH, BUILD_PATH)
    # stage_7_parametrize.normalize_kidneys(BUILD_PATH)
    # stage_7_parametrize.display_all_normalizations(BUILD_PATH)
    # stage_7_parametrize.build_spectral_feature_vectors(BUILD_PATH)
    # stage_7_parametrize.build_binary_feature_vectors(BUILD_PATH)
    # stage_7_parametrize.principal_component_analysis(BUILD_PATH)
    # stage_7_parametrize.display_subject_clusters(BUILD_PATH, DATA_PATH)
    # stage_7_parametrize.display_kidney_clusters(BUILD_PATH, DATA_PATH)
    # stage_5_measure.measure_normalized_mask_shape(BUILD_PATH)
    stage_7_parametrize.build_dice_correlation_matrix(BUILD_PATH)
    # stage_7_corr.build_all_correlation_matrices(BUILD_PATH)

    # NOTE: Display by site
    # stage_7_parametrize.display_all_normalizations(BUILD_PATH, 'Controls')
    # for site in ['Bordeaux', 'Exeter', 'Leeds', 'Bari', 'Sheffield', 'Turku']:
    #     stage_7_parametrize.display_all_normalizations(BUILD_PATH, 'Patients', site)

    # stage_6_archive.normalizations(BUILD_PATH, ARCHIVE_PATH)




if __name__ == '__main__':

    # build_canvas()
    preprocess_rsf()
    # run_preprocessing()
    # run_manual_section()
    # run_postprocessing()
    # run_shape_analysis()


    

