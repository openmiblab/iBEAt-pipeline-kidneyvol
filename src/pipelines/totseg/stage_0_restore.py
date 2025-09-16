import os

import dbdicom as db


def dixons(local_path, shared_path, group, site=None):
    datapath = os.path.join(local_path, 'dixon', 'stage_2_data')
    archivepath = os.path.join(shared_path, "dixon", "stage_2_data")
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, 'Controls')
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, 'Patients', site)
        sitearchivepath = os.path.join(archivepath, 'Patients', site)
    db.restore(sitearchivepath, sitedatapath)



