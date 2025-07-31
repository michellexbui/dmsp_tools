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
import matplotlib as mpl
import pandas as pd
from netCDF4 import Dataset
import os

def main():
    # inputs here!!
    # -------------
    satname = str(input("Input satellite name as a string: e.g. $ f17\n")) 
    strdates = list(input("Input list of dates in YYYYMMDD format, split by a comma: e.g. $ 20100404,20100405,20100406,20100407,20100408,20100409\n").split(',')) 

    # plot polar plots
    #polar_plots(satname, strdates)

    # hemispheric power
    HPI = {'time': [], 'hpi' : []}

    for date_str in strdates:
        dirpath = find_SSUSI_path(date_str,satname)

        for filename in os.listdir(dirpath):
            # check if .NC file
            if filename.endswith('.NC') != True: 
                continue # skips 1 iteration 

            # get data
            # --------
            SSUSI_PATH = os.path.join(dirpath, filename) 
            ssusi=Dataset(SSUSI_PATH)

            data_point = float(ssusi['HEMISPHERE_POWER_NORTH'][0].item())
            data_time_sec = float(ssusi['TIME'][0])
            data_time_dt = dt.datetime.strptime(date_str, '%Y%m%d') + dt.timedelta(seconds=data_time_sec)
            
            print(f'At TIME: {data_time_dt}, HPI: {data_point} GW');

            # assign HPI to timestamp
            # -----------------------
            HPI['time'].append(data_time_dt)
            HPI['hpi'].append(data_point/2)


    print(HPI)

    fig, ax = plt.subplots()
    ax.plot(HPI['time'], HPI['hpi'],'o-')
    ax.set_title('DMSP-SSUSI Total Hemispheric Power')

    ax.set_ylim(bottom=0.0)
    ax.set_ylabel('GigaWatts')
    ax.set_xlabel(f'{HPI['time'][0].strftime('%Y-%m-%d')} to {HPI['time'][-1].strftime('%Y-%m-%d')}')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))

    plt.savefig('HPI.png')



def polar_plots(sat_name, str_of_dates):
    for date_str in str_of_dates: 
        # setup
        # -----
        # directory check and data path
        path_energyflux = 'figures/energyflux/' + date_str + '/'
        dir_exist(path_energyflux)
        path_meanenergy = 'figures/meanenergy/' + date_str + '/'
        dir_exist(path_meanenergy)
        
        #find path to SSUSI file
        dirpath = find_SSUSI_path(date_str,sat_name)
    
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
                image[fp] = np.nan
                energy_n[fp] = np.nan
            
                # get timestamp
                starttime = ssusi.STARTING_TIME
                stoptime = ssusi.STOPPING_TIME
                if starttime[:7] != stoptime[:7]:
                    ut[ut > 20] = ut[ut > 20]-24 # limits data within 1 day

                yyyy = int(stoptime[:4])
                ddd = int(stoptime[4:7])
                date = pd.Timestamp(yyyy, 1, 1)+pd.Timedelta(ddd-1, 'D')
                timestamp =  date+pd.Timedelta(np.nanmean(ut[image==image]),'h')
                event_dt = timestamp.to_pydatetime()   
            
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
                    unit = r'$mW/m^2$'
                elif "MEAN" in plottype:
                    title = "Mean Energy Patterns"
                    cmap_str = "plasma"
                    plotpath = path_meanenergy
                    maxi = 6 # MXB Q: same what does this mean physically?
                    mini = 0 # MXB Note: I used Mukhopadhyay et al 2022 Fig 8b max/mins for this
                    unit = r'$keV$'
    
                # plot
                plot_SSUSI(dataplot,mlat,mlt,maxi,mini,event_dt,title, cmap_str, unit)
                plotname = event_dt.strftime('%Y%m%d_%H%M') + '_energyflux.png'

                # output
                # ------
                plt.savefig(plotpath + plotname, dpi=150)


def plot_SSUSI(image,mlat,mlt,maxi,mini,time_stamp,name,cmap_str,unit_str):
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
    ax.set_title(name + "\n" + timestamp_str + '\n \n')

    fig.colorbar(hs, cax=ax_cbar, shrink=0.3, label=unit_str)

    return

def find_SSUSI_path(date_str, sat_name):
    # year and day-of-year
        year = date_str[0:4]   
        datetime_Ymd = dt.datetime.strptime(date_str, '%Y%m%d')
        datetime_doy = datetime_Ymd.timetuple().tm_yday
        doy = f'{datetime_doy:03d}'

        # SSUSI directory path
        path_to_dir = f'/backup/Data/ssusi/data/ssusi.jhuapl.edu/dataN/{sat_name}/apl/edr-aur/{year}/{doy}/'

        return path_to_dir

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

if __name__=="__main__":
    main()
