import os

import dbdicom as db


def archive_autosegmentation(group, site=None):
    datapath = os.path.join(os.getcwd(), 'build', 'kidneyvol', 'stage_1_segment')
    archivepath = os.path.join("G:\\Shared drives", "iBEAt Build", 'kidneyvol', 'stage_1_segment')
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, "Controls") 
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, "Patients", site) 
        sitearchivepath = os.path.join(archivepath, "Patients", site)
    db.archive(sitedatapath, sitearchivepath)


def archive_edits(group, site=None):
    datapath = os.path.join(os.getcwd(), 'build', 'kidneyvol', 'stage_3_edit')
    archivepath = os.path.join("G:\\Shared drives", "iBEAt Build", 'kidneyvol', 'stage_3_edit')
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, "Controls") 
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, "Patients", site) 
        sitearchivepath = os.path.join(archivepath, 'Patients', site)
    db.archive(sitedatapath, sitearchivepath)
    