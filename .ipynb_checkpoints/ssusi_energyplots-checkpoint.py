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
# - fig : 'figures/energyflux/YYYYMMDD/YYYYMMDD_HHMM_energyflux.png'
# - fig : 'figures/meanenergy/YYYYMMDD/YYYYMMDD_HHMM_meanenergy.png'
# 
# FUNCTIONS 
# - plot_SSUSI : makes polar plots
# - dir_exist : checks if a path exists, and creates if not
# ====================


import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import os

def main():
    # inputs here!!
    # -------------
    sat_name = 'f17'
    str_dates = ['20100404','20100405','20100406',
                 '20100407','20100408','20100409']

    for date_str in str_dates: 
        # setup
        # -----
        # directory check and data path
        path_energyflux = 'figures/energyflux/' + date_str + '/'
        dir_exist(path_energyflux)
        path_meanenergy = 'figures/meanenergy/' + date_str + '/'
        dir_exist(path_meanenergy)
            
        # year and day-of-year
        year = date_str[0:4]   
        date_fmt = '%Y%m%d'
        datetime_Ymd = dt.datetime.strptime(date_str, date_fmt)
        datetime_doy = datetime_Ymd.timetuple().tm_yday
        doy = f'{datetime_doy:03d}'
        # SSUSI directory path
        dirpath = f'/backup/Data/ssusi/data/ssusi.jhuapl.edu/dataN/{sat_name}/apl/edr-aur/{year}/{doy}/'
    
        # search files & plot
        # -------------------
        for filename in os.listdir(dirpath):
            # check if .NC file
            if filename.endswith('.NC') != True:
                continue # skips 1 iteration

            # get data
            SSUSI_PATH = os.path.join(dirpath, filename) 
            ssusi=Dataset(SSUSI_PATH)

            nodatavalue = ssusi.NO_DATA_IN_BIN_VALUE  # value in a no data bin
            ut = ssusi['UT_N'][:]
            vartmp = np.array(ssusi['DISK_RADIANCEDATA_INTENSITY_NORTH'])

            # make plots for energy flux and mean energy
            for plottype in ['ENERGY_FLUX_NORTH_MAP',
                             'ELECTRON_MEAN_NORTH_ENERGY_MAP']:
                image = vartmp[4,:,:]
                energy_n = np.array(ssusi[plottype])
                fp = (ut == nodatavalue)
                energy_n[fp] = np.nan
                image[fp] = np.nan
            
                # get timestamp
                starttime = ssusi.STARTING_TIME
                stoptime = ssusi.STOPPING_TIME
                yyyy = int(stoptime[:4])
                ddd = int(stoptime[4:7])
                date = pd.Timestamp(yyyy, 1, 1)+pd.Timedelta(ddd-1, 'D')
                timestamp = date+pd.Timedelta(np.nanmean(ut[image==image]),'h')
            
                if starttime[:7] != stoptime[:7]:
                    ut[ut > 20] = ut[ut > 20]-24 # MXB Q: what does this do?
            
                # set up plot
                dataplot = energy_n
                mlat = np.array(ssusi['LATITUDE_GEOMAGNETIC_GRID_MAP']) 
                mlt = np.array(ssusi['MLT_GRID_MAP'])

                # titles / formatting
                if 'FLUX' in plottype:
                    title = "Energy Flux Patterns"
                    plotpath = path_energyflux
                    cmap_str = "magma"
                    maxi = 15 # MXB Q: what does this mean physically
                    mini = 0  # MXB Note: I used Mukhopadhyay et al 2022 Fig 8a max/mins for this
                elif "MEAN" in plottype:
                    title = "Mean Energy Patterns"
                    cmap_str = "plasma"
                    plotpath = path_meanenergy
                    maxi = 6 # MXB Q: same what does this mean physically?
                    mini = 0 # MXB Note: I used Mukhopadhyay et al 2022 Fig 8b max/mins for this
    
                # plot
                plot_SSUSI(dataplot,mlat,mlt,maxi,mini,timestamp,title, cmap_str)
                event_dt = timestamp.to_pydatetime()
                plotname = event_dt.strftime('%Y%m%d_%H%M') + '_energyflux.png'

                # output
                # ------
                plt.savefig(plotpath + plotname, dpi=150)

def plot_SSUSI(image,mlat,mlt,maxi,mini,time_stamp,name,cmap_str):
    #
    # OBJECTIVE 
    # - creates polar plots
    #
    # INPUT TYPES 
    # - array : image, mlat, mlt
    # - int : maxi, mini
    # - pandas.Timestamp : time_stamp
    # - str : name, cmap_str 
    #
    # OUTPUT TYPES 
    # - fig : pyplot
    # 
    
    fig=plt.figure()
    plt.subplots_adjust(left = 0.15,   right=0.85,
                        wspace = 0.03, hspace = 0.03)
    ax = fig.add_subplot(1,1,1,polar=True)

    # plot polar map
    theta = mlt*15.0*np.pi/180.0-np.pi/2
    rad = 90.0-mlat

    hs=ax.scatter(theta, rad, c=image,       s=0.5,
                              vmin=mini,     vmax=maxi,
                              cmap=cmap_str, alpha=0.6)
    levels = [0.0,10,20,30,40]

    ax.set_rticks(levels)
    ax.set_rmax(40.0)
    ax.set_yticklabels(['','','','','50'])
    ax.set_rlabel_position(22.5)
    ax.set_xticks(np.arange(0,2*np.pi,np.pi/2.0))
    ax.set_xticklabels(['06','12', '18', '00'])
    ax.grid(True)
    ax.set_title(name + "\n" + str(time_stamp))

    fig.colorbar(hs,ax=ax,shrink=0.7)

    return

def dir_exist(path):
    #
    # OBJECTIVE 
    # - checks if a path exists. if not, creates that path.
    #
    # INPUT 
    # - str : path     # e.g. 'figures/energyflux/20100405/'
    # 
    
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f'Created dir: {path}')
    else:
        print(f'Dir exists: {path}')
    
# main loop
# =========
if __name__=="__main__":
    main()
