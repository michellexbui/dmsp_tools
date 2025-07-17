import pandas as pd
import os, glob
import h5py
import xarray as xr
import numpy as np
from apexpy import Apex
from scipy.signal import find_peaks # to find the mlat extrema
import matplotlib.pyplot as plt


# inputs
sat_name = 'F16' #DMSP satellite number
lat_threshold = 60 # if we want to make high-latitude plots

date_str = '20110926'               # date in yyyymmdd format
DL_PATH = f'/backup/Data/DMSP/{year}/'  # Folder with DMSP files


# Date formatting
date_req = pd.to_datetime(date_str)      # convert to datetime object
date_str_req = str(date_req.year) + str(date_req.month).rjust(2,'0') + str(date_req.day).rjust(2,'0')
year = date_str[0:4]




def get_dmsp_data(DL_PATH,date_str_req,sat_name):
    # Read DMSP data 
    dfd = read_hdf5_file(DL_PATH, date_str_req, file_type='drifts', sat_list=[sat_name])
    dfp = read_hdf5_file(DL_PATH, date_str_req, file_type='precip', sat_list=[sat_name])
    # Merge precip and drifts files
    df = pd.merge(dfd, dfp, on='dtime', how='outer', suffixes=('_dfd', '_dfp'))
    df = clean_identical_columns(df) # Remove identical variables
    for col in ['mlt', 'mlong', 'glon']: # Fix cyclical parameters
        newdf = fix_col(df, col)
    return newdf



def read_hdf5_file(DL_PATH, date_str, file_type='drifts', # also works for 'precip'
                   sat_list=None):
    ds = {}

    if file_type == 'drifts':
        fl_nm_DMSP = os.path.join(DL_PATH, f"dms_{date_str}_s1")
    elif file_type == 'precip':
        all_files = glob.glob(os.path.join(DL_PATH, f"dms_{date_str}_*"))
        fl_nm_DMSP = [f for f in all_files if "s1" not in f]
        
    found_files = glob.glob(fl_nm_DMSP) if isinstance(fl_nm_DMSP, str) else fl_nm_DMSP
    
    for ifile in found_files:
    
        #print('Reading HDF5 file:', ifile)
         # Extract satellite number
        satnum = ifile[ifile.rfind('_')+1:]
        satnum = satnum[:satnum.find('.')-1]
        satkey = 'F' + satnum
        #print(satkey)
        if file_type == 'drifts':
            satkey = satkey[0:-1]
        elif file_type == 'precip':
            satkey = satkey
        # Skip if sat_list is provided and this satkey is not in it
        if sat_list is not None and satkey not in sat_list:
            continue
        
        if file_type == 'drifts':
            h5cols = ['year', 'month', 'day', 'hour', 'min', 'sec', 'recno',
                      'kindat', 'kinst', 'ut1_unix', 'ut2_unix',
                      "gdlat", "glon", "gdalt", "sat_id", "mlt", "mlat", "mlong",
                      "ne", "hor_ion_v", "vert_ion_v", "bd", "b_forward", "b_perp",
                      "diff_bd", "diff_b_for", "diff_b_perp"]
    
            # Read the HDF5 file into a pandas DataFrame
            # Faster than Dask & Easier to manipulate than xarray since it is flat
            with h5py.File(ifile, 'r') as file:
                if file_type == 'drifts':
                    data_group = file['Data']
                    data_array = data_group['Table Layout'][:]
                df = pd.DataFrame({name: data_array[name] for name in h5cols})
    
            df['dtime'] = pd.to_datetime(df.ut1_unix, unit='s')
            df = df.set_index('dtime')
            # remove columns year -> ut2_unix (first 11 cols)
            df = df[h5cols[11:]]
    
        elif file_type == 'precip':
            ds = xr.open_dataset(ifile, engine='h5netcdf')
            ds['dtime'] = ('timestamps', pd.to_datetime(ds.timestamps.values, unit='s'))
            ds = ds.swap_dims({'timestamps':'dtime'}).drop_vars('timestamps')
            df = ds.drop_dims('ch_energy').to_dataframe()

        #ds[satkey] = df

    return df



# remove or clean the identical columns from merged file in precip and drifts files
def clean_identical_columns(df, atol=1e-10):
    for col in df.columns:
        if col.endswith('_dfd'):
            base = col[:-4]
            col_dfp = base + '_dfp'

            if col_dfp in df.columns:
                try:
                    # Compute difference
                    diff = df[col_dfp] - df[col]
                    mean_diff = np.nanmean(diff)

                    if np.abs(mean_diff) < atol:
                        df[base] = df[col]  # or df[col_dfp], they're the same
                        df.drop(columns=[col, col_dfp], inplace=True)
                        print(f" Merged: {base} (mean diff = {mean_diff:.2e})")
                    else:
                        print(f" Couldn't merge: {base} (mean diff = {mean_diff:.2e})")

                except Exception as e:
                    print(f"{base}: Could not compare ({e})")

    return df



# Fix the data artefacts due to cyclic coordinate interpolation in glon,mlon, and mlt
def fix_col(df, colname):
    ts = pd.date_range(df.index.min().date(),
                       df.index.max(), freq='1 min')
    ts = [t for t in ts if t in df.index]
    threshold = 0.2 if colname == 'mlt' else 2 # 2 for mlon and lon; 0.2 for mlt
    for iTime in range(len(ts) - 1):
        t0, t1 = ts[iTime], ts[iTime + 1]
        window = df.loc[t0:t1, colname]
        if (np.abs(np.diff(window)) > threshold).sum() > 4:
            if colname == 'mlt':
                seg  = df.loc[t0:t1]
                dt0  = seg.index[0].to_pydatetime()
                t = seg.index

                mlon = seg['mlong'].to_numpy()
                alt  = seg['gdalt'].to_numpy()
                apex = Apex(date=dt0)
                new_mlt = apex.mlon2mlt(mlon, t.to_numpy(), alt)
                df.loc[t0:t1, colname] = new_mlt

            else:
                v0 = df.at[t0, colname] 
                v1 = df.at[t1, colname]
                n = len(window)
                df.loc[t0:t1, colname] = np.linspace(v0, v1, num=n)

    return df


def get_dmsp_peak_times(newdf,lat_threshold):
    #DMSP Peak Detection
    mlat_abs = np.abs(newdf.mlat)
    satpeaks, _ = find_peaks(mlat_abs)
    near_polar_peaks = satpeaks[mlat_abs.iloc[satpeaks] > lat_threshold]
    # Get peak indices and times
    peak_times_str = newdf.index[near_polar_peaks]
    # Convert peak time strings to datetime objects
    peak_times = [t.to_pydatetime() for t in peak_times_str]
    return peak_times
newdf = get_dmsp_data(DL_PATH,date_str_req,sat_name)
peak_times = get_dmsp_peak_times(newdf,lat_threshold)
plt.plot(newdf['el_i_flux'])