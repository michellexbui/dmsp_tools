'''
Objective: Access DMSP SSUSI EDR products

Examples
--------
'''


import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import netCDF4
from netCDF4 import Dataset

import xarray as xr

import os
import pickle

def main():
    # just get the pickle : see example in pickle_ssusiday
    # get ssusi maps : see example in plot_SSUSImaps
        # name your desired inputs
    strdates = ['20180129']                 
    strsats = ['f17']
    sourcename = 'cdaweb'

    # plot ur maps!
    plot_SSUSImaps(strsats, strdates, sourcename)     
    
    # # define your desired inputs
    # date_strlist = ['20110926','20110805','20111025','20120308', '20120423', '20120930']
    # sat_name = 'f17'
    # sourcename = 'cdaweb'

    # # loop through your dates to get your desired pickles
    # for date_str in date_strlist:
    
    #     # find the path of SSUSI EDR aurora data
    #     dirpath = find_SSUSI_path(date_str,sat_name,sourcename)

    #     # pickle the data
    #     pickled_day = pickle_ssusiday(date_str, dirpath)

def calc_HP(ssusi_day):
    '''
    Objective: Calculatethe hemispheric power for a day of SSUSI observations

    Parameters 
    ----------
    ssusi_day : pickle
        Pickle object containing one day of SSUSI EDR aurora data

    Returns
    -------
    HP : dict
        dictionary containing:
        'time'          = list of datetime 
        'dA'            = list of dA of the integral [km]
        'hp_calc'       = list of integrated energy flux over spherical area [GW] 
        'hp_ssusiguvi'  = list of hemispheric power provided in the SSUSI EDR aurora data [GW] 

    Examples
    --------
    HP = calc_HP(pickled_ssusiday)

    '''
    # Earth radius [km]
    r_E = 6378.0 * 1000

    # create dict to store these vals
    HP = {'time' : [] ,             # datetime 
          'dA' : [],                # [km] dA of the integral
          'hp_calc' : [],           # [GW] integrated energy flux over spherical area 
          'hp_ssusiguvi' : []}      # [GW] hemispheric power provided in the SSUSI EDR aurora data 

    # now iterate for all times in the ssusi day
    for ts in ssusi_day.keys():

        # access one timestamp
        ts_dt = dt.datetime.strptime(ts, '%Y%j%H%M%S') # move this line to the pickle so the pickle keys are datetimes instead fo strings
        ssusi = ssusi_day[ts]
        energy_flux = np.array(ssusi['ENERGY_FLUX_NORTH_MAP'])  # units: ergs/s/cm^2
        
        # set up and solve the integral
        integrand = np.zeros((len(energy_flux[:,0])-1,len(energy_flux[0,:])-1))
        for j in range(0,len(energy_flux[:,0])-1):
            for i in range(0,len(energy_flux[0,:])-1):
                r = r_E + np.array(ssusi['HME_NORTH'])[j,i]*1000
                del_glat = np.abs(glat_try[j,i+1]-glat_try[j,i])*np.pi/180                           
                del_glon = np.abs(glon_try[j+1,i]-glon_try[j,i])*np.pi/180
                if 0.0 <= energy_flux[j,i]: # <= maxi
                    integrand[j,i] = energy_flux[j,i] * 0.001 * r**2 * np.sin(90-glat_try[j,i]*np.pi/180) * del_glat * del_glon                 # convert ergs/s/cm^2 to W/m^2

        # append vals to our dictionary
        HP['time'].append(ts_dt)
        HP['hp_calc'].append(np.sum(integrand)/1.0e9)
        HP['hp_ssusiguvi'].append(float(ssusi['HEMISPHERE_POWER_NORTH']))
        HP['dA'].append(r*del_glat*del_glon)

    return HP

def pickle_ssusiday(date_str, sat_name, dirpath):
    '''
    Objective: Serialize '.nc' data files to Python pickle objects, which allow for efficient storage and de-serialization.

    Parameters
    ----------
    date_str : str   
        Desired date as a string, formatted as 'YYYYMMDD' 
        where   YYYY is the four-digit year, 
                MM is a zero-padded month, and 
                DD is a zero-padded day
    dirpath : str
        Path to SSUSI EDR aurora data .nc files for one day

    Returns
    -------
    pickled_ssusiday : Python pickle
        Store a day of SSUSI EDR aurora data into a single pickle. 

    Examples
    --------
    # define your desired inputs
    date_strlist = ['20150623','20150317','20120309','20130317']
    sat_name = 'f17'
    sourcename = 'cdaweb'

    # loop through your dates to get your desired pickles
    for date_str in date_strlist:
    
        # find the path of SSUSI EDR aurora data
        dirpath = find_SSUSI_path(date_str,sat_name,sourcename)

        # pickle the data
        pickled_day = pickle_ssusiday(date_str,sat_name, dirpath)

    '''
    # MXB NOTE: should a pickle have one dataset (i.e. 1 timestamp in 1 day) or should a pickle have multiple datasets within a day?
    # whole day pickle
    pklname = f'pickles/ssusi_{sat_name}_{date_str}.pkl'
    date_dataset = {}

    # each file in the pickle jar
    for filename in os.listdir(dirpath):
        # check if .NC file
        if filename.endswith('.NC') != True and filename.endswith('.nc') != True: 
            continue # skips 1 iteration 

        # get data
        SSUSI_PATH = os.path.join(dirpath, filename) 
        dataset_temp = xr.open_dataset(SSUSI_PATH)
        dataset = dataset_temp.load()

        # append str day/time to dataset, formatted as 'YYYDDDHHMMSS'
        date_dataset.update({dataset.STARTING_TIME : dataset})

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

def plot_SSUSImaps(strlist_of_sats, strlist_of_dates, sourcename='cdaweb'):
    '''
    Objective: Given a list of satellites and dates, create and save maps of energy flux and mean energy.

    Parameters
    ----------
    strlist_of_sats : list of string objects
        List of desired satellites
        e.g. ['f16', 'f17', 'f18']
    strlist_of_dates : list of string objects
        List of desired dates
        Desired date as a string, formatted as 'YYYYMMDD' 
        where   YYYY is the four-digit year, 
                MM is a zero-padded month, and 
                DD is a zero-padded day
        e.g. ['20100405', '20220203']
    sourcename : str
        Desired data source. 
        Default is 'cdaweb': download 'nc' from CDAWeb public server.
        If running locally on mia, data can be sourced from mia backup data

    Returns
    -------
    saves figures at
        dmsp_tools/figures/energyflux/YYYYMMDD/*.png
        dmsp_tools/figures/meanenergy/YYYYMMDD/*.png


    Examples
    --------
    # name your desired inputs
    strdates = ['20100405']                 
    strsats = ['f17','f18']
    sourcename = 'cdaweb'

    # plot ur maps!
    plot_SSUSImaps(strsats, strdates, sourcename) 

    '''
    for sat_name in strlist_of_sats:
        # loop for each intended satellite
        for date_str in strlist_of_dates: 
            # setup
            # -----
            # check if a dir for that date exists 
            dir_exist(f'figures/energyflux/{date_str}/'); dir_exist(f'figures/energyflux/{date_str}/{sat_name}')
            dir_exist(f'figures/meanenergy/{date_str}/') ; dir_exist(f'figures/meanenergy/{date_str}/{sat_name}')
            
            # find path to SSUSI file & pickle it
            # -----------------------------------
            dirpath = find_SSUSI_path(date_str,sat_name,sourcename)
            pickled_ssusi = pickle_ssusiday(date_str, dirpath)

            # use ssusi pickle
            # ----------------
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
                    # MXB note: i should be returning fig/ax objects

            if sourcename == 'cdaweb':
                # make space
                os.system(f'rm -r uplodat/{date_str}/')

def plot_polar(image,mlat,mlt,maxi,mini,time_stamp,name,cmap_str,unit_str, sat_name):
    '''
    Objective: Create well-formatted polar plots for SUSSI applications.

    Parameters
    ----------
    image : numpy array
        2-D array of values to plot
    mlat, mlt : numpy array
        Magnetic lat and local time associated with `image`.
    mini, maxi : float
        Minimum and maximum values
    time_stamp : datetime.datetime
        Starting date and time in UT of recorded data
    name : string
        Title of plot
    cmap_str : string
        String with Matplotlib choice of colormap. See: https://matplotlib.org/stable/users/explain/colors/colormaps.html 
    unit_str : string
        String of data units

    Returns
    -------
    MXB note: i should be returning fig/ax objects

    Examples
    --------
    # create the polar plot 
    plot_polar(dataplot,mlat,mlt,maxi,mini,event_dt,title, cmap_str, unit, sat_name)

    # name and save the plot
    plotname = 'plotname.png'
    plt.savefig(plotname)

    '''

    fig = plt.figure()
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

def find_SSUSI_path(date_str, sat_name, sourcename):
    '''
    Objective: Download SSUSI EDR (Environmental Data Record) aurora data from 'cdaweb' public server if run anywhere or 'mia' server if run locally on mia
    > Downloaded '.nc' files are located in uplodat/{date_str}/ 

    Parameters
    ----------
    date_str : str   
        Desired date as a string, formatted as 'YYYYMMDD' 
        where   YYYY is the four-digit year, 
                MM is a zero-padded month, and 
                DD is a zero-padded day
    sat_name : str
        Desired satellite name ('f17', 'f18', etc.) as a string.
        See here for data availability: https://docs.google.com/spreadsheets/d/1QyxeKCH3AZUILgoSASgSuJIv4_F9cR3c1689ooKb8dk/edit?gid=0#gid=0 
    sourcename : str
        Desired data source. 
        If 'cdaweb', download 'nc' from CDAWeb public server and save to uplodat/{date_str}
        If running locally on 'mia', data can be sourced from mia's backup repo
        
    Returns
    -------
    path_to_dir : str
        Path to SSUSI EDR aurora data .nc files
    
    Examples
    --------
    # source the EDR aurora data .nc files
    dirpath = find_SSUSI_path(date_str,sat_name,sourcename)

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

def find_val(array_to_search, target_val):
    '''
    Objective: Find closest-to-desired value in a 2D data array

    Parameters
    ----------
    array_to_search : 2D array
        Array of data that you would like to search
    target_val : float
        Desired value that you are searching for in a large array

    Returns
    -------
    row, col : int
        Row and column index for the desired value
    
    Examples
    --------
    # find location of a specific value
    maxi = 20.0
    row, col = find_val(np.array(ssusi['ENERGY_FLUX_NORTH_MAP']), maxi) 

    '''

    diff_arr = np.abs(array_to_search - target_val)
    flattened_ind = np.argmin(diff_arr)
    row, col = np.unravel_index(flattened_ind, diff_arr.shape)

    return row, col

# ==================
# functions that are more useful for the case study comparison
# i should probably make a separate .py for case study comparison
# ==================

def add_circle_boundary(ax):
    '''
    Objective: Compute a circle in axes coordinates. Allows us to use a circular boundary for a Cartopy map.
                Reference for this method: Cartopy always circular stereo example
                https://scitools.org.uk/cartopy/docs/v0.15/examples/always_circular_stereo.html 

    Parameters
    ----------
    ax : Matplotlib axis object

    Returns
    -------
    None

    Examples
    --------
    # initiate figure
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1,2,1, projection=ccrs.NorthPolarStereo())

    # add content here

    # finalize figure
    ax.set_title('Title of the Plot')
    ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
    add_circle_boundary(ax)

    '''
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    return


def solar_position(date_UT):
    '''
    Objective: Calculate the geographic longitude and latitude point on Earth which the Sun is directly overhead.

    Parameters
    ----------
    date_UT : Datetime object
        A datetime object in Universal Time.

    Returns
    -------
    slon, slat : float
        Geographic longitude and latitude at which the Sun is directly overhead the Earth.

    Examples
    --------
    date_UT = datetime.datetime(2010, 4, 5, 9, 30, 0)
    slon, slat = solar_position(date_UT)

    '''
    # extract datetime info
    year = date_UT.year
    month = date_UT.month
    day = date_UT.day
    hour = date_UT.hour
    minute = date_UT.minute
    second = date_UT.second
    
    # 1. Calculate a Julian Date
    # ==========================
    # using Vallado, David (2007) "Fundamentals of Astrodynamics and Applications" Algorithm 14
    if month < 3:
        month += 12
        year -= 1

    B = 2 - year // 100 + (year // 100) // 4
    C = ((second / 60 + minute) / 60 + hour) / 24

    JD = (int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5 + C)

    # 2. Calculate the Solar Position
    # ===============================
    # centuries from January 2000
    T_UT1 = (JD(date) - 2451545.0) / 36525

    # solar longitude (deg)
    lambda_M_sun = (280.460 + 36000.771 * T_UT1) % 360

    # solar anomaly (deg)
    M_sun = (357.5277233 + 35999.05034 * T_UT1) % 360

    # ecliptic longitude
    lambda_ecliptic = (lambda_M_sun + 1.914666471 * np.sin(np.deg2rad(M_sun)) +
                       0.019994643 * np.sin(np.deg2rad(2 * M_sun)))

    # obliquity of the ecliptic (epsilon in Vallado's notation)
    epsilon = 23.439291 - 0.0130042 * T_UT1

    # declination of the sun
    slat = np.rad2deg(np.arcsin(np.sin(np.deg2rad(epsilon)) *
                                     np.sin(np.deg2rad(lambda_ecliptic))))

    # Greenwich mean sidereal time (seconds)
    theta_GMST = (67310.54841 +
                  (876600 * 3600 + 8640184.812866) * T_UT1 +
                  0.093104 * T_UT1**2 -
                  6.2e-6 * T_UT1**3)
    # Convert to degrees
    theta_GMST = (theta_GMST % 86400) / 240

    # Right ascension calculations
    numerator = (np.cos(np.deg2rad(epsilon)) *
                 np.sin(np.deg2rad(lambda_ecliptic)) /
                 np.cos(np.deg2rad(slat)))
    denominator = (np.cos(np.deg2rad(lambda_ecliptic)) /
                   np.cos(np.deg2rad(slat)))

    alpha_sun = np.rad2deg(np.arctan2(numerator, denominator))

    # longitude is opposite of Greenwich Hour Angle (GHA)
    # GHA == theta_GMST - alpha_sun
    slon = -(theta_GMST - alpha_sun)
    if slon < -180:
        slon += 360

    return (slon, slat)    

def dir_exist(path):
    '''
     Objective: Checks if a path to a directory exists. If the directory does not exist, then the function creates that directory.
    
    Parameters
    ----------
    path : str    
        Path to desired directory

    Returns
    -------
    None

    Example
    -------
    path = 'figures/energyflux/20100405/
    dir_exist(path)

    '''
    
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f'Created dir: {path}')
    else:
        print(f'Dir exists: {path}')

if __name__=="__main__":
    main()

