# ====================
# ssusi_energyplots.py
#
# ABOUT ME
# - Using DMSP-SSUSI data files (.NC), we can plot polar energy flux and mean energy maps for a day.
#   DMSP-SSUSI files are located in mia (defined in main as 'dirpath').
# - You need to plug in the satellite name and a list of dates.
#   The main loop will search for the SSUSI data files for that date and plot pretty polar pictures.
#   These pics are saved in 'figures/' + [data type i.e. meanenergy or energyflux] + /YYYYMMDD
# 
# MAIN INPUTS 
# - sat_name : str               # sat_name: pick from ['f16','f17','f18']
# - str_dates : [str, str, ...]  # where each str is formatted YYYYMMDD
# 
# MAIN OUTPUTS
# 
# FUNCTIONS 
# - pickle_ssusi : unpacks a ssusi file
# - plot_polar : makes polar plots
# - dir_exist : checks if a path exists, and creates if not
# ====================


import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import netCDF4
import xarray as xr
from netCDF4 import Dataset
import os
import aacgmv2
import pickle

def main():
    # this is sacred keep this
    strdates = ['20100405'] #['20110805','20110926','20111024','20120307','20120423','20120616','20120715','20120930','20121007','20121113','20130317','20130531','20130628','20220203','20220204','20240510']
    strsats = ['f17'] #,'f18']
    sourcename = 'mia'
    plot_SSUSImaps_wpickle(strsats, strdates, sourcename) 


# ==================== HPI...whats wrong...

def HPI_plots(strdates,sat_name,sourcename):
    # hemispheric power
    HPI = {'time': [], 'hpi' : []}

    for date_str in strdates:
        dirpath = find_SSUSI_path(date_str,sat_name,sourcename)

        for filename in os.listdir(dirpath):
            # check if .NC file
            if filename.endswith('.NC') != True: 
                continue # skips 1 iteration 

            # get data
            # --------
            SSUSI_PATH = os.path.join(dirpath, filename) 
            ssusi=Dataset(SSUSI_PATH)

            data_point = float(ssusi['HEMISPHERE_POWER_NORTH'][0])
            data_time_sec = float(ssusi['TIME'][0])
            data_time_dt = dt.datetime.strptime(date_str, '%Y%m%d') + dt.timedelta(seconds=data_time_sec)

            # assign HPI to timestamp
            # -----------------------
            HPI['time'].append(data_time_dt)
            HPI['hpi'].append(data_point)

    HPI_df = pd.DataFrame.from_dict(HPI)
    HPI_df = HPI_df.sort_values(by='time') # sort chronologically

    print(HPI_df)
    
    left_date = HPI_df['time'][1]
    print(left_date)
    right_date = HPI_df['time'][HPI_df.index[-1]]
    print(right_date)

    fig, ax = plt.subplots()
    ax.plot(HPI_df['time'], HPI_df['hpi'],'o-',color='red',label='DMSP-SSUSI')
    ax.set_title(f'Total Hemispheric Power ({sat_name})')

    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=left_date, right=right_date)
    ax.set_ylabel('GigaWatts')
    ax.set_xlabel(f'{left_date.strftime('%Y-%m-%d')} to {right_date.strftime('%Y-%m-%d')}')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))

    plt.grid(linestyle='--', color='gray', alpha=0.7)
    plt.legend()
    plt.savefig(f'figures/hemisphericpower/{left_date.strftime('%Y%m%d_%H%M')}-{right_date.strftime('%Y%m%d_%H%M')}-{sat_name}-HPI.png')


# ==================== PICKLING HERE
# yipee we incorporated pickles

def pickle_ssusiday(date_str, dirpath):
    # MXB NOTE: should a pickle have one dataset (i.e. 1 timestamp in 1 day) or should a pickle have multiple datasets within a day?
    # whole day pickle
    pklname = f'ssusi_{date_str}.pkl'
    date_dataset = {}

    # each file in the pickle jar
    for filename in os.listdir(dirpath):
        # check if .NC file
        if filename.endswith('.NC') != True and filename.endswith('.nc') != True: 
            continue # skips 1 iteration 

        # get data
        SSUSI_PATH = os.path.join(dirpath, filename) 
        dataset = xr.open_dataset(SSUSI_PATH)

        # append str day/time to dataset, formatted as 'YYYMMMHHMMSS'
        date_dataset.update({dataset.STOPPING_TIME : dataset})

    # write into a pickle
    with open(pklname, 'wb') as f:
        pickle.dump(date_dataset, f)

    # MXB NOTE: do i need to close the file?
    f.close()
    
    # open as a read only
    with open(pklname, 'rb') as file:
        pickled_ssusiday = pickle.load(file)

    # return the data to be used
    return pickled_ssusiday

def pickle_1ssusi(dirpath, filename):
    # OBJECTIVE: pickle 1 dataset
    SSUSI_PATH = os.path.join(dirpath, filename) 
    dataset = xr.open_dataset(SSUSI_PATH)

    str_timestamp = dataset.STOPPING_TIME
    pklname = f'ssusi_{str_timestamp}.pkl'

    with open(pklname, 'wb') as f:
        pickle.dump(dataset, f)

    f.close() # MXB NOTE: do i need this? will it close auto?

    with open(pklname, 'rb') as file:
        pickled_1ssusi = pickle.load(file)
    
    return pickled_1ssusi

def plot_SSUSImaps(strlist_of_sats, strlist_of_dates,sourcename='mia'):
    for sat_name in strlist_of_sats:
        # loop for each intended satellite
        for date_str in strlist_of_dates: 
            # setup
            # -----
            # check if a dir for that date exists 
            dir_exist(f'figures/energyflux/{date_str}/'); dir_exist(f'figures/energyflux/{date_str}/{sat_name}')
            dir_exist(f'figures/meanenergy/{date_str}/') ; dir_exist(f'figures/meanenergy/{date_str}/{sat_name}')
            
            # find path to SSUSI file
            # -----------------------
            dirpath = find_SSUSI_path(date_str,sat_name,sourcename)

            #===
            pickled_ssusi = pickle_ssusiday(date_str, dirpath)

            for eventtime in pickled_ssusi.keys():
                ssusi = pickled_ssusi[eventtime]

                nodatavalue = ssusi.NO_DATA_IN_BIN_VALUE  # value in a no data bin
                ut = ssusi['UT_N'][:]
                vartmp = np.array(ssusi['DISK_RADIANCEDATA_INTENSITY_NORTH'])

                # make plots for energy flux and mean energy
                for plottype in ['ENERGY_FLUX_NORTH_MAP',
                                'ELECTRON_MEAN_NORTH_ENERGY_MAP']:
                    image = vartmp[4,:,:]
                    energy_n = np.array(ssusi[plottype])
                    fp = (ut == nodatavalue)
                    image[fp] = np.nan
                    energy_n[fp] = np.nan
                
                    # MXB NOTE: CHANGE THIS. Use starttime str => datetime object
                    # get timestamp
                    starttime = ssusi.STARTING_TIME
                    stoptime = ssusi.STOPPING_TIME

                    yyyy = int(stoptime[:4])
                    ddd = int(stoptime[4:7])
                    date = pd.Timestamp(yyyy, 1, 1)+pd.Timedelta(ddd-1, 'D')

                    event_dt = dt.datetime.strptime(starttime, '%Y%j%H%M%S')
                
                    # set up plot
                    dataplot = energy_n
                    mlat = np.array(ssusi['LATITUDE_GEOMAGNETIC_GRID_MAP']) 
                    mlt = np.array(ssusi['MLT_GRID_MAP'])

                    # titles / formatting
                    if 'FLUX' in plottype:
                        title = "Energy Flux Patterns"
                        plotpath = f'figures/energyflux/{date_str}/{sat_name}/'
                        cmap_str = "magma"
                        maxi = 15 # MXB Q: what does this mean physically
                        mini = 0  # MXB Note: I used Mukhopadhyay et al 2022 Fig 8a max/mins for this
                        unit = r'$mW/m^2$'
                        name_type = 'ENERGYFLUX'
                    elif 'MEAN' in plottype:
                        title = "Mean Energy Patterns"
                        cmap_str = "plasma"
                        plotpath = f'figures/meanenergy/{date_str}/{sat_name}/'
                        maxi = 6 # MXB Q: same what does this mean physically?
                        mini = 0 # MXB Note: I used Mukhopadhyay et al 2022 Fig 8b max/mins for this
                        unit = r'$keV$'
                        name_type = 'MEANENERGY'
        
                    # plot
                    plot_polar(dataplot,mlat,mlt,maxi,mini,event_dt,title, cmap_str, unit, sat_name)
                    plotname = f'{event_dt.strftime('%Y%m%d_%H%M')}-{sat_name}-{name_type}.png'

                    # output
                    # ------
                    plt.savefig(plotpath + plotname, dpi=150)
                    plt.close() 

            if sourcename == 'cdaweb':
                # make space
                os.system(f'rm -r uplodat/{date_str}/')

def plot_polar(image,mlat,mlt,maxi,mini,time_stamp,name,cmap_str,unit_str, sat_name):
    #
    # OBJECTIVE 
    # - creates polar plots
    #
    # INPUT TYPES 
    # - array : image, mlat, mlt
    # - int : maxi, mini
    # - datetime.datetime : time_stamp
    # - str : name, cmap_str, unit_str
    #
    # OUTPUT TYPES 
    # - fig : pyplot
    # 
    
    fig=plt.figure()
    plt.subplots_adjust(bottom = 0.2,  top = 0.8,
                        wspace = 0.03, hspace = 0.03)
    
    ax = fig.add_subplot(1,1,1,polar=True)
    ax_cbar = ax.inset_axes([1, 0, 0.05, 0.3])

    # plot polar map
    theta = mlt*15.0*np.pi/180.0-np.pi/2
    rad = 90.0-mlat

    hs=ax.scatter(theta, rad, c=image,       s=0.5,
                              vmin=mini,     vmax=maxi,
                              cmap=cmap_str, alpha=0.6)
    
    levels = [0.0,10,20,30,40]
    ax.set_rticks(levels)
    ax.set_rmax(40.0)
    ax.set_rlabel_position(22.5)

    ax.set_yticklabels(['N','80','70','60','']) # lat labels
    ax.tick_params(axis='y', labelcolor='gray')

    ax.set_xticks(np.arange(0,2*np.pi,np.pi/2.0)) # MLT labels
    ax.set_xticklabels(['06','12', '18', '00 MLT'])

    ax.grid(True)

    timestamp_str = time_stamp.strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title(f'{name} ({sat_name})\n {timestamp_str} \n \n')
    #ax.set_title(name + '(' + sat_name + ')' + "\n" + timestamp_str + '\n \n')

    fig.colorbar(hs, cax=ax_cbar, shrink=0.3, label=unit_str)

    return

def find_SSUSI_path(date_str, sat_name, sourcename='mia'):
    '''
    OBJECTIVE
    - find data from two different sources: mia or cdaweb. 
    - if mia, output is the path to the mia data
    - if cdaweb, it will wget cdaweb data and save into uplodat/{date_str}/. then output is the path to the uplodat/{date_str}/ data

    INPUT
    - date_str : str    
        e.g. '20100405' : YYYYMMDD format
    - sat_name : str    
        e.g. 'f17'
    - sourcename : str  
        e.g. 'mia' or 'cdaweb'

    OUTPUT
    - path_to_dir : str
        e.g. 'uplodat/{date_str}
    '''
    # year and day-of-year
    year = date_str[0:4]   
    datetime_Ymd = dt.datetime.strptime(date_str, '%Y%m%d')
    datetime_doy = datetime_Ymd.timetuple().tm_yday
    doy = f'{datetime_doy:03d}'

    if sourcename == 'mia':
        # SSUSI directory path
        path_to_dir = f'/backup/Data/ssusi/data/ssusi.jhuapl.edu/dataN/{sat_name}/apl/edr-aur/{year}/{doy}/'
    elif sourcename == 'cdaweb':
        # CHECK IF uplodat/ and uplodat/{date_str}/ exists
        dir_exist('uplodat/')
        dir_exist(f'uplodat/{date_str}')

        # then upload data
        urlstr = f'https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmsp{sat_name}/ssusi/data/edr-aurora/{year}/{doy}/'
        # wget index.html 
        os.system(f'wget -P uplodat/{date_str}/ {urlstr}') 

        # read index.html for filenames
        filename = f'uplodat/{date_str}/index.html' ; files = []
        with open(filename, 'r', encoding='utf-8') as f:
            html_content = f.read()

        lines = html_content.splitlines() 
        for eachline in lines:
            index_start = eachline.find('dmsp')
            index_end = eachline.find('.nc')

            if len(eachline[index_start:index_end+3]) > 0:
                files.append(eachline[index_start:index_end+3])

        # wget files into uplodat/{date_str}
        for file in files:
            os.system(f'wget -P uplodat/{date_str} {urlstr}{file}')
        
        # success u have the files!
        path_to_dir = f'uplodat/{date_str}'

    return path_to_dir

def dir_exist(path):
    '''
     OBJECTIVE 
    - checks if a path to a directory exists. if not, creates that directory.
    
    INPUT 
    - path : str    e.g. 'figures/energyflux/20100405/'
                    path to the directory you are intersted in viewing

    OUTPUT (none)
    '''
    
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f'Created dir: {path}')
    else:
        print(f'Dir exists: {path}')

if __name__=="__main__":
    main()

