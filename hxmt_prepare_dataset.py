import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from astropy.io import fits
from tatpulsar.utils.functions import cal_event_gti

FEATURE = ['SAT_X', 'SAT_Y', 'SAT_Z', 'SAT_ALT', 'SAT_LON', 'SAT_LAT', 'ELV', 'DYE_ELV',
           'DYE_ELV', 'ANG_DIST', 'COR', 'T_SAA', 'TN_SAA', 'SUN_ANG', 'MOON_ANG', 'SUNSHINE',
           "TH_HE_PHODet_0", "TH_HE_PHODet_1", "TH_HE_PHODet_2", "TH_HE_PHODet_3", "TH_HE_PHODet_4",
           "TH_HE_PHODet_5", "TH_HE_PHODet_6", "TH_HE_PHODet_7", "TH_HE_PHODet_8", "TH_HE_PHODet_9",
           "TH_HE_PHODet_10", "TH_HE_PHODet_11", "TH_HE_PHODet_12", "TH_HE_PHODet_13", "TH_HE_PHODet_14",
           "TH_HE_PHODet_15", "TH_HE_PHODet_16", "TH_HE_PHODet_17",
           "TH_HE_PM_0", "TH_HE_PM_1", "TH_HE_PM_2",
           'Cnt_PM_0', 'Cnt_PM_1', 'Cnt_PM_2',
           "HE_Cnt_VetoDet_0", "HE_Cnt_VetoDet_1", "HE_Cnt_VetoDet_2", "HE_Cnt_VetoDet_3", "HE_Cnt_VetoDet_4",
           "HE_Cnt_VetoDet_5", "HE_Cnt_VetoDet_6", "HE_Cnt_VetoDet_7", "HE_Cnt_VetoDet_8", "HE_Cnt_VetoDet_9",
           "HE_Cnt_VetoDet_10", "HE_Cnt_VetoDet_11", "HE_Cnt_VetoDet_12", "HE_Cnt_VetoDet_13", "HE_Cnt_VetoDet_14",
           "HE_Cnt_VetoDet_15", "HE_Cnt_VetoDet_16", "HE_Cnt_VetoDet_17",
           'HV_HE_PHODet_0', 'HV_HE_PHODet_1', 'HV_HE_PHODet_2', 'HV_HE_PHODet_3', 'HV_HE_PHODet_4',
           'HV_HE_PHODet_5', 'HV_HE_PHODet_6', 'HV_HE_PHODet_7', 'HV_HE_PHODet_8', 'HV_HE_PHODet_9',
           'HV_HE_PHODet_10', 'HV_HE_PHODet_11', 'HV_HE_PHODet_12', 'HV_HE_PHODet_13', 'HV_HE_PHODet_14',
           'HV_HE_PHODet_15', 'HV_HE_PHODet_16', 'HV_HE_PHODet_17']




def organize_HE_smfov_data(datadir, obsid, outfile=None):
    """
    load and prepare the dataset for HE small FoV by observation ID
    """

    # CONSTANT
    chnum = 256
    channel_bins_blind  = [f'BLIND_COUNTS_CHANNEL_{i}' for i in range(chnum)]
    channel_bins_normal = [f'NORMAL_COUNTS_CHANNEL_{i}' for i in range(chnum)]

    # Check Files
    evtfile = glob.glob(os.path.join(datadir, f"{obsid}_HE_screen_RAW.fits")                )
    EHKfile = glob.glob(os.path.join(datadir, f"HXMT_{obsid}_EHK_FFFFFF_V?_L1P.FITS")       )
    attfile = glob.glob(os.path.join(datadir, f"HXMT_{obsid}_Att_FFFFFF_V?_L1P.FITS")       )
    orbfile = glob.glob(os.path.join(datadir, f"HXMT_{obsid}_Orbit_FFFFFF_V?_L1P.FITS")     )
    THfile  = glob.glob(os.path.join(datadir, f"HXMT_{obsid}_HE-TH_FFFFFF_V?_L1P.FITS")     )
    PMfile  = glob.glob(os.path.join(datadir, f"HXMT_{obsid}_HE-PM_FFFFFF_V?_L1P.FITS")     )
    ACfile  = glob.glob(os.path.join(datadir, f"HXMT_{obsid}_HE-Cnts_FFFFFF_V?_L1P.FITS")   )
    HVfile  = glob.glob(os.path.join(datadir, f"HXMT_{obsid}_HE-HV_FFFFFF_V?_L1P.FITS")     )
    file_recipes = [evtfile, EHKfile, attfile, orbfile, THfile, PMfile, ACfile, HVfile]
    for file in file_recipes:
        if file == []:
            return
        else:
            print(file)
    evtfile = evtfile[-1]
    EHKfile = EHKfile[-1]
    attfile = attfile[-1]
    orbfile = orbfile[-1]
    THfile  = THfile[-1]
    PMfile  = PMfile[-1]
    ACfile  = ACfile[-1]
    HVfile  = HVfile[-1]
        #print(f"Checking files...{file}")
        #if not os.path.exists(file):
        #    print(f"{evtfile} does not exists")
        #    return
        #    #raise IOError(f"{evtfile} does not exists")

    # Load Event file
    hdulist = fits.open(evtfile)
    time_event = hdulist['HEEvt'].data['TIME']
    pi_event   = hdulist['HEEvt'].data['PI']
    detid_event= hdulist['HEEvt'].data['Det_ID']

    # Load EHK file
    hdulist = fits.open(EHKfile)
    time_ehk = hdulist['EHK'].data['Time']
    SAT_X    = hdulist['EHK'].data['X']
    SAT_Y    = hdulist['EHK'].data['Y']
    SAT_Z    = hdulist['EHK'].data['Z']
    SAT_ALT  = hdulist['EHK'].data['SAT_ALT']
    SAT_LON  = hdulist['EHK'].data['SAT_LON']
    SAT_LAT  = hdulist['EHK'].data['SAT_LAT']
    ELV      = hdulist['EHK'].data['ELV']
    DYE_ELV  = hdulist['EHK'].data['DYE_ELV']
    ANG_DIST = hdulist['EHK'].data['ANG_DIST']
    COR      = hdulist['EHK'].data['COR']
    T_SAA    = hdulist['EHK'].data['T_SAA']
    TN_SAA   = hdulist['EHK'].data['TN_SAA']
    SUN_ANG  = hdulist['EHK'].data['SUN_ANG']
    MOON_ANG = hdulist['EHK'].data['MOON_ANG']
    SUNSHINE = hdulist['EHK'].data['SUNSHINE']

    # Load Temperature File
    hdulist = fits.open(THfile)
    time_th = hdulist['TH_HE_PHODet'].data['Time']
    TH_HE_PHODet_all = [] # Powswich Temperature
    for i in range(18):
        TH_HE_PHODet_all.append(hdulist['TH_HE_PHODet'].data[f'TH_HE_PHODet_{int(i)}P'])
    TH_HE_PM_all = [] # Particle Monitor detector temperature
    for i in range(3):
        TH_HE_PM_all.append(hdulist['TH_HE_PM'].data[f'TH_HE_PM_{int(i)}'])

    # Load PM file
    hdulist = fits.open(PMfile)
    time_pm = hdulist['HE_Cnt_PM'].data['Time']
    Cnt_PM_0= hdulist['HE_Cnt_PM'].data['Cnt_PM_0']
    Cnt_PM_1= hdulist['HE_Cnt_PM'].data['Cnt_PM_1']
    Cnt_PM_2= hdulist['HE_Cnt_PM'].data['Cnt_PM_2']

    # Load Anti-Coincidence Detectors
    hdulist = fits.open(ACfile)
    time_ac = hdulist['HE_Cnt_VetoDet'].data['Time']
    VetoDet_all = []
    for i in range(18):
        VetoDet_all.append(hdulist['HE_Cnt_VetoDet'].data[f'Cnt_VetoDet_{int(i)}'])

    # Load HV file
    hdulist = fits.open(HVfile)
    time_hv = hdulist['HE_HV_PHODet'].data['Time']
    HV_PHODet_all = []
    for i in range(18):
        HV_PHODet_all.append(hdulist['HE_HV_PHODet'].data[f'HV_PHODet_{int(i)}'])

    ## Check if those time array in different File are identical
    if _are_identical(time_ehk, time_th, time_pm, time_hv):
        print("Data is valid...")

    # Start to Organize the dataset
    num_sec = time_ehk.size
    cnt_blind = np.zeros((num_sec, chnum)) # np.zeros(chnum)]*time_ehk.size
    cnt_normal = np.zeros((num_sec, chnum)) #[np.zeros(chnum)]*time_ehk.size
    time_start_event = time_event.min()
    GTIs = cal_event_gti(time_event, tgap=1)

    for index, one_sec in enumerate(tqdm(time_ehk, desc="Formating Data for each second")):
        if one_sec < time_start_event:continue
        if not _is_in_good_interval(one_sec, GTIs):continue
        time_mask = (time_event>=one_sec)&(time_event<=one_sec+1)

        # Blind detector spectrum
        pi_blind_one_sec = pi_event[time_mask&(detid_event==16)]
        cnt_blind_one_sec, channel_blind_one_sec = np.histogram(pi_blind_one_sec,
                                                                np.arange(0, chnum+1, 1))
        channel_blind_one_sec = channel_blind_one_sec[:-1]
#        if np.all(cnt_blind_one_sec==0):continue ## If spectrum is empty, skip
        cnt_blind[index] = cnt_blind_one_sec
        # Normal detector spectrum
        pi_normal_one_sec = pi_event[time_mask&(detid_event!=16)]
        cnt_normal_one_sec, channel_normal_one_sec = np.histogram(pi_normal_one_sec,
                                                                  np.arange(0, chnum+1, 1))
        channel_normal_one_sec = channel_normal_one_sec[:-1]
#        if np.all(cnt_normal_one_sec==0):continue ## If spectrum is empty, skip
        cnt_normal[index] = cnt_normal_one_sec


    data = {
            # Incorporate MET
            'MET' : time_ehk,

            ## Satellite Location & Pointing Angle
            "SAT_X": SAT_X,
            "SAT_Y": SAT_Y,
            "SAT_Z": SAT_Z,
            "SAT_ALT": SAT_ALT,
            "SAT_LON": SAT_LON,
            "SAT_LAT": SAT_LAT,

            ## Solar's Position
            "SUN_ANG"  : SUN_ANG,
            "MOON_ANG" : MOON_ANG,
            "SUNSHINE" : SUNSHINE,

            ## Geomagnetic
            "ELV"      : ELV,
            "DYE_ELV"  : DYE_ELV,
            "ANG_DIST" : ANG_DIST,
            "COR"      : COR,
            "T_SAA"    : T_SAA,
            "TN_SAA"   : TN_SAA,

            ### PM counts
            "Cnt_PM_0": Cnt_PM_0,
            "Cnt_PM_1": Cnt_PM_1,
            "Cnt_PM_2": Cnt_PM_2,

            }

    ### Anti-Coincidence Detector Counts
    for det in range(18):
        data[f"Cnt_VetoDet_{det}"] = VetoDet_all[det]

    # Telescope STATUS
    for det in range(18):
        data[f"TH_HE_PHODet_{det}"] = TH_HE_PHODet_all[det] # Temperature of Phoswich
        data[f"HV_PHODet_{det}"] = HV_PHODet_all[det]
    for det in range(3):
        data[f"TH_HE_PM_{det}"] = TH_HE_PM_all[det]

    # Spectra
    for chn_i, bin_name in enumerate(channel_bins_blind):
        data[bin_name] = np.asarray(cnt_blind[:, chn_i])
    for chn_i, bin_name in enumerate(channel_bins_normal):
        data[bin_name] = np.asarray(cnt_normal[:, chn_i])

    ## Exclude those data pieces out of GTI (GTI was the minimum filtering on the raw data)
    ## If spectrum of normal detector or blind detector is empty, then skip the corresponding data slice
    mask = np.all(cnt_blind == np.zeros(chnum), axis=1) + np.all(cnt_normal == np.zeros(chnum), axis=1)
    masked_data = {}
    for key, value in data.items():
        masked_data[key] = value[~mask]

    if outfile:
        df = pd.DataFrame(masked_data)
        df.to_csv(outfile, index=False)
    return masked_data


# Check if all arrays are identical
def _are_identical(*arrays):
    first_array = arrays[0]
    return all(np.array_equal(first_array, array) for array in arrays[1:])

def _is_in_good_interval(Ti, GTIs):
    for gti in GTIs:
        if gti[0] <= Ti <= gti[1]:
            return True
    return False

def _process_obsid(args):
    obsid, datadir, outfile = args
    print(f"Starting {obsid}")  # Debugging output
    #outfile = f"out/dataset/dataset_{obsid}.csv"
    if os.path.exists(outfile):
        return f"{obsid} already processed."
    organize_HE_smfov_data(datadir, obsid, outfile=outfile)
    return f"{obsid} processed."

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensures compatibility on various platforms

    __version__ = '1.0'


    ehkfiles = glob.glob("/Volumes/hxmt/DataHub/HXMT/BKGTraining/HE/HXMT_P*EHK*")[:]
    obsids = [os.path.basename(x).split('_')[1] for x in ehkfiles]
    datadir = "/Volumes/hxmt/DataHub/HXMT/BKGTraining/HE"

    # Create a list of arguments for the process_obsid function
    args = [(obsid, datadir, f"out/dataset/dataset_{obsid}_v{__version__}.csv") for obsid in obsids]

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(_process_obsid, args), total=len(obsids), desc="Processing ObsIDs"))

#    for obsid in obsids:
#        print(obsid)
#        outfile = f"out/dataset/dataset_{obsid}_v{__version__}.csv"
#        if os.path.exists(outfile):continue
#        data = organize_HE_smfov_data(datadir,
#                                      obsid,
#                                      outfile=outfile)

