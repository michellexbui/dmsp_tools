# dmsp_tools
DMSP-SSUSI (Special Sensor Ultraviolet Spetrograph Imagers) is an instrument used to observe aurora from about 840 km. Here's a working toolkit for exploring DMSP-SSUSI data.

## Plot mean energy and energy flux patterns
All you need to input is the satellite name (f16, f17, f18) and a list dates in YYYYMMDD,YYYYMMDD,... format. Once you run the code, your figures will appear in figures/meanenergy/YYYYMMDD/ and figures/energyflux/YYYYMMDD. 

Here's how you could run the code to generate your own plots:

```
$ python3 ssusi_energyplots.py
> Input satellite name as a string: e.g. $ f17
$ f17
> Input list of dates in YYYYMMDD format, split by a comma: e.g. $ 20100404,20100405,20100406,20100407,20100408,20100409
$ 20100404,20100405,20100406,20100407,20100408,20100409
```

This will generate figures located in `figures/meanenergy/YYYYMMDD/` and `figures/energyflux/YYYYMMDD`, where YYYYMMDD is a directory for each date in your date list.


**Here are some resources for DMSP data:** \
Boston College DSMP Survey Plots: https://dmsp.bc.edu - survey plots, descriptions of all instruments, useful links \
CEDAR Madrigal: https://cedar.openmadrigal.org/ \
CDAWeb: https://cdaweb.gsfc.nasa.gov/  \
APL SSUSI:
https://ssusi.jhuapl.edu - apparently the govt nuked this site ): 
