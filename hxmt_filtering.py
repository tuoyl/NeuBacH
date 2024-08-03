import os
import glob
"""
Perform the basic filtering for HXMT data
(only those mandatory filtering, e.g. spikes, anti-coincidence events ...)
"""

def HXMT_HE_filter(datadir, obsid):
        """
        hegtigen hvfile="/hxmt/work/HXMT-DATA/1L/A01/P0101293/P0101293001/P010129300101-20171102-01-01/HE/HXMT_P010129300101_HE-HV_FFFFFF_V1_L1P.FITS"
        tempfile="/hxmt/work/HXMT-DATA/1L/A01/P0101293/P0101293001/P010129300101-20171102-01-01/HE/HXMT_P010129300101_HE-TH_FFFFFF_V1_L1P.FITS"
        pmfile="/hxmt/work/HXMT-DATA/1L/A01/P0101293/P0101293001/P010129300101-20171102-01-01/HE/HXMT_P010129300101_HE-PM_FFFFFF_V1_L1P.FITS"
        outfile="/hxmt/work/USERS/tuoyl/BKGmodel_training/dataset/rawdata/P010129300101-20171102-01-01/P010129300101_HE_gti.fits"
        ehkfile="/hxmt/work/HXMT-DATA/1L/A01/P0101293/P0101293001/P010129300101-20171102-01-01/AUX/HXMT_P010129300101_EHK_FFFFFF_V1_L1P.FITS"
        defaultexpr="NONE" expr="ELV>10&&COR>8&&SAA_FLAG==0&&TN_SAA>300&&T_SAA>300&&ANG_DIST<=0.04"
        pmexpr="" clobber="yes" history="yes"
hescreen evtfile="/hxmt/work/USERS/tuoyl/BKGmodel_training/dataset/rawdata/P010129300101-20171102-01-01/P010129300101_HE_pi.fits"
gtifile="/hxmt/work/USERS/tuoyl/BKGmodel_training/dataset/rawdata/P010129300101-20171102-01-01/P010129300101_HE_gti.fits"
outfile="/hxmt/work/USERS/tuoyl/BKGmodel_training/dataset/rawdata/P010129300101-20171102-01-01/P010129300101_HE_screen.fits"
userdetid="0-17" eventtype=1 anticoincidence="yes" starttime=0 stoptime=0 minPI=0 maxPI=255 clobber="yes" history="yes
(base) 18:28:16@ ~/Work/Background/BackgroundTraining/HXMT_training$ ls /Volumes/hxmt/DataHub/HXMT/BKGTraining/HE/*P0101293001*
        """
        gtigen_cmd = f'hegtigen hvfile="{datadir}/HXMT_{obsid}_HE-HV_FFFFFF_V1_L1P.FITS" '+\
                     f'tempfile="{datadir}/HXMT_{obsid}_HE-TH_FFFFFF_V1_L1P.FITS" '+\
                     f'pmfile="{datadir}/HXMT_{obsid}_HE-PM_FFFFFF_V1_L1P.FITS" ' +\
                     f'outfile="{datadir}/{obsid}_HE_gti_RAW.fits" '+\
                     f'ehkfile="{datadir}/HXMT_{obsid}_EHK_FFFFFF_V1_L1P.FITS" ' + \
                     f'defaultexpr="NONE" expr="ELV>0&&COR>0&&SAA_FLAG==0&&TN_SAA>0&&T_SAA>0&&ANG_DIST<=0.04" ' +\
                     'pmexpr="" clobber="yes" history="yes"'

        screen_cmd = f'hescreen evtfile="{datadir}/{obsid}_HE_pi.fits" ' +\
                     f'gtifile="{datadir}/{obsid}_HE_gti_RAW.fits" ' +\
                     f'outfile="{datadir}/{obsid}_HE_screen_RAW.fits" ' + \
                     f'userdetid="0-17" eventtype=1 anticoincidence="yes" '+\
                     'starttime=0 stoptime=0 minPI=0 maxPI=255 clobber="yes" history="yes" '

        print(gtigen_cmd)
        print(screen_cmd)

if __name__ == "__main__":
    ehkfiles = glob.glob("/Volumes/hxmt/DataHub/HXMT/BKGTraining/HE/HXMT_P*EHK*")[:]
    obsids = [os.path.basename(x).split('_')[1] for x in ehkfiles]
    datadir = "/Volumes/hxmt/DataHub/HXMT/BKGTraining/HE"

    for obsid in obsids:
        HXMT_HE_filter(datadir, obsid)
