from pipelines.kidneyvol import (
    stage_0_restore, 
    stage_1_segment, 
    stage_2_display,
    stage_3_edit,
    stage_4_display,
    stage_5_measure,
    stage_6_archive
)


def run():
    
   site = 'Exeter'
   stage_0_restore.dixons('Patients', site)
   stage_0_restore.segmentations('Patients', site)
   stage_5_measure.measure('Patients', site)

   for site in ['Leeds', 'Bordeaux']:
      stage_0_restore.dixons('Patients', site)
      stage_0_restore.segmentations('Patients', site)
      stage_4_display.mosaic('Patients', site)
      stage_5_measure.measure('Patients', site)


