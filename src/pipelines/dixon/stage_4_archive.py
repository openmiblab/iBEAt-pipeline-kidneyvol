import os


import dbdicom as db

datapath = os.path.join(os.getcwd(), 'build', 'dixon', 'stage_2_data')
archivepath = target_dir = os.path.join("G:\\Shared drives", "iBEAt Build", "dixon", "stage_2_data")


def archive_clean_dixons(site):
    if site=='Controls':
        sitedatapath = os.path.join(datapath, 'Controls')
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, 'Patients', site)
        sitearchivepath = os.path.join(archivepath, 'Patients', site)
    db.archive(sitedatapath, sitearchivepath)


if __name__=='__main__':

    # archive_clean_dixons('Leeds')
    # archive_clean_dixons('Sheffield')
    # archive_clean_dixons('Bari')
    archive_clean_dixons('Bordeaux')
    # archive_clean_dixons('Exeter')
    # archive_clean_dixons('Controls')
    

