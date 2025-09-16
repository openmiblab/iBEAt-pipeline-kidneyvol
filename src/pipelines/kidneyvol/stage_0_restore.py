import os
import logging

import dbdicom as db


def dixons(group, site=None):
    datapath = os.path.join(os.getcwd(), 'build', 'dixon', 'stage_2_data')
    archivepath = os.path.join("G:\\Shared drives", "iBEAt Build", "dixon", "stage_2_data")
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, 'Controls')
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, 'Patients', site)
        sitearchivepath = os.path.join(archivepath, 'Patients', site)
    db.restore(sitearchivepath, sitedatapath)


def segmentations(group, site=None):
    datapath = os.path.join(os.getcwd(), 'build', 'kidneyvol', 'stage_1_segment')
    archivepath = os.path.join("G:\\Shared drives", "iBEAt Build", "kidneyvol", "stage_1_segment")
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, 'Controls')
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, 'Patients', site)
        sitearchivepath = os.path.join(archivepath, 'Patients', site)
    db.restore(sitearchivepath, sitedatapath)

    datapath = os.path.join(os.getcwd(), 'build', 'kidneyvol', 'stage_3_edit')
    archivepath = os.path.join("G:\\Shared drives", "iBEAt Build", "kidneyvol", "stage_3_edit")
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, 'Controls')
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, 'Patients', site)
        sitearchivepath = os.path.join(archivepath, 'Patients', site)
    db.restore(sitearchivepath, sitedatapath)


if __name__=='__main__':

    dixons('Bari')

