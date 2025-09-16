"""
Task: Build a clean database with all pre- and post contrast Dixon scans.

dixon_1_download: Download from XNAT

dixon_1_data: Build a clean database
- standard series file organisation
- standard patient IDs
- standard sequence names and numbers
- correct fat water swap
- List best series among repetitions

dixon_3_check: perform checks on the database
- visualise fat water swaps per site
- visualise duplicates for easy selection of best
- Build summary csv with all sequences and counts
"""

from pipelines.dixon import (
    stage_1_download,
    stage_2_data,
    stage_3_check,
    stage_4_archive
)

def run():
    # stage_1_download.sheffield_patients()
    # stage_2_data.sheffield()
    # stage_2_data.exeter_patients('Baseline')
    # stage_2_data.exeter_patients('Followup')
    # stage_3_check.check_fatwater_swap('Exeter')
    stage_4_archive.archive_clean_dixons('Sheffield')

