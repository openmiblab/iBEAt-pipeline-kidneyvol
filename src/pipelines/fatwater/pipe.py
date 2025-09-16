from pipelines.fatwater import (
   stage_0_restore, 
   stage_1_waterdom,
   stage_2_trainingdata,
   stage_3_trainmodel,
)


def run():


   all_sites = ['Bordeaux', 'Bari', 'Leeds', 'Sheffield', 'Turku', 'Exeter']
   
   # # stage_0_restore.dixons('Controls')
   # # for site in all_sites:
   # #    stage_0_restore.dixons('Patients', site)

   # stage_1_waterdom.compute('Controls')
   # for site in all_sites:
   #    stage_1_waterdom.compute('Patients', site)
   
   #stage_2_trainingdata.generate()


   # stage_3_trainmodel.preprocess()
   stage_3_trainmodel.train(True)





