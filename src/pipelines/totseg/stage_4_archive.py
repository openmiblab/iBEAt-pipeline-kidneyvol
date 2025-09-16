import os

import dbdicom as db

from utils.files import copy_new_files



def autosegmentation(local_path, shared_path, group, site=None):
    datapath = os.path.join(local_path, 'totseg', 'stage_1_segment')
    archivepath = os.path.join(shared_path, 'totseg', 'stage_1_segment')
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, "Controls") 
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, "Patients", site) 
        sitearchivepath = os.path.join(archivepath, "Patients", site)
    db.archive(sitedatapath, sitearchivepath)


def displays(local_path, shared_path, group, site=None):
    datapath = os.path.join(local_path, 'totseg', 'stage_2_display')
    archivepath = os.path.join(shared_path, 'totseg', 'stage_2_display')
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, "Controls") 
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, "Patients", site) 
        sitearchivepath = os.path.join(archivepath, "Patients", site)
    copy_new_files(sitedatapath, sitearchivepath)
    