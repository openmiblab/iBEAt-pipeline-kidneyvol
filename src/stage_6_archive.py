import os

import dbdicom as db


def autosegmentation(build_path, archive_path, group, site=None):
    datapath = os.path.join(build_path, 'kidneyvol', 'stage_1_segment')
    archivepath = os.path.join(archive_path, 'kidneyvol', 'stage_1_segment')
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, "Controls") 
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, "Patients", site) 
        sitearchivepath = os.path.join(archivepath, "Patients", site)
    db.archive(sitedatapath, sitearchivepath)


def edits(build_path, archive_path, group, site=None):
    datapath = os.path.join(build_path, 'kidneyvol', 'stage_3_edit')
    archivepath = os.path.join(archive_path, 'kidneyvol', 'stage_3_edit')
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, "Controls") 
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, "Patients", site) 
        sitearchivepath = os.path.join(archivepath, 'Patients', site)
    db.archive(sitedatapath, sitearchivepath)

def normalizations(build_path, archive_path):
    datapath = os.path.join(build_path, 'kidneyvol', 'stage_7_normalized')
    archivepath = os.path.join(archive_path, 'kidneyvol', 'stage_7_normalized')
    db.archive(datapath, archivepath)
    