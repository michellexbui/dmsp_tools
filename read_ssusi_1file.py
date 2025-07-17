import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

def main():

    ifile0 = '/backup/Data/ssusi/data/ssusi.jhuapl.edu/dataN/f18/apl/edr-aur/2017/001/'
    ifile0 = ifile0 + 'PS.APL_V0105S027CB0006_SC.U_DI.A_GP.F18-SSUSI_PA.APL-EDR-AURORA_DD.20170101_SN.37156-00_DF.NC'

    ssusi=Dataset(ifile0)
    MLat = np.array(ssusi['LATITUDE_GEOMAGNETIC_GRID_MAP'])
    MLT = np.array(ssusi['MLT_GRID_MAP'])
    nodatavalue = ssusi.NO_DATA_IN_BIN_VALUE  # value in a no data bin
    ut = ssusi['UT_N'][:]
    vartmp = np.array(ssusi['DISK_RADIANCEDATA_INTENSITY_NORTH'])
    image = vartmp[4,:,:]
    energyfluxn = np.array(ssusi['ENERGY_FLUX_NORTH_MAP'])
    fp = (ut == nodatavalue)
    energyfluxn[fp] = np.nan
    image[fp] = np.nan

    starttime = ssusi.STARTING_TIME
    stoptime = ssusi.STOPPING_TIME
    yyyy = int(stoptime[:4])
    ddd = int(stoptime[4:7])
    date = pd.Timestamp(yyyy, 1, 1)+pd.Timedelta(ddd-1, 'D')

    if starttime[:7] != stoptime[:7]:
        ut[ut > 20] = ut[ut > 20]-24

    dataplot = energyfluxn
    mlat = MLat
    mlt = MLT

    maxi = 10
    mini = 0

    timestamp = date+pd.Timedelta(np.nanmean(ut[image==image]),'h')

    get_one_image(dataplot,mlat,mlt,maxi,mini)

    plt.show()

    return


def get_one_image(image,mlat,mlt,maxi,mini):

    fig=plt.figure()
    plt.subplots_adjust(left = 0.15,
            right=0.85,
            wspace = 0.03,hspace = 0.03)
    ax = fig.add_subplot(1,1,1,polar=True)

    #plot polar map
    theta = mlt*15.0*np.pi/180.0-np.pi/2
    rad = 90.0-mlat

    hs=ax.scatter(theta,rad,c=image,
            s=0.5,vmin=mini,vmax=maxi,
            cmap='plasma',alpha=0.6)
    levels = [0.0,10,20,30,40]
    ax.set_rticks(levels)
    ax.set_rmax(40.0)
    ax.set_yticklabels(['','','','','50'])
    ax.set_rlabel_position(22.5)
    ax.set_xticks(np.arange(0,2*np.pi,np.pi/2.0))
    ax.set_xticklabels(['06','12', '18', '00'])
    ax.grid(True)

    fig.colorbar(hs,ax=ax,shrink=0.7)


    return

if __name__=="__main__":

    main()
