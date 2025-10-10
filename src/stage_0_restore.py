import os

import dbdicom as db


def dixons(archivepath, buildpath, group, site=None):
    local_buildpath = os.path.join(buildpath, 'dixon', 'stage_2_data')
    local_archivepath = os.path.join(archivepath, "dixon", "stage_2_data")
    if group == 'Controls':
        sitebuildpath = os.path.join(local_buildpath, 'Controls')
        sitearchivepath = os.path.join(local_archivepath, 'Controls')
    else:
        sitebuildpath = os.path.join(local_buildpath, 'Patients', site)
        sitearchivepath = os.path.join(local_archivepath, 'Patients', site)
    db.restore(sitearchivepath, sitebuildpath)


def segmentations(archivepath, buildpath, group, site=None):
    local_buildpath = os.path.join(buildpath, 'kidneyvol', 'stage_1_segment')
    local_archivepath = os.path.join(archivepath, "kidneyvol", "stage_1_segment")
    if group == 'Controls':
        sitebuildpath = os.path.join(local_buildpath, 'Controls')
        sitearchivepath = os.path.join(local_archivepath, 'Controls')
    else:
        sitebuildpath = os.path.join(local_buildpath, 'Patients', site)
        sitearchivepath = os.path.join(local_archivepath, 'Patients', site)
    db.restore(sitearchivepath, sitebuildpath)


def edited_segmentations(archivepath, buildpath, group, site=None):
    local_buildpath = os.path.join(buildpath, 'kidneyvol', 'stage_3_edit')
    local_archivepath = os.path.join(archivepath, "kidneyvol", "stage_3_edit")
    if group == 'Controls':
        sitebuildpath = os.path.join(local_buildpath, 'Controls')
        sitearchivepath = os.path.join(local_archivepath, 'Controls')
    else:
        sitebuildpath = os.path.join(local_buildpath, 'Patients', site)
        sitearchivepath = os.path.join(local_archivepath, 'Patients', site)
    db.restore(sitearchivepath, sitebuildpath)

def normalized_kidneys(archivepath, buildpath):
    buildpath = os.path.join(buildpath, 'kidneyvol', 'stage_7_normalized')
    archivepath = os.path.join(archivepath, "kidneyvol", "stage_7_normalized")
    db.restore(archivepath, buildpath)



