import os, sys
import math
import atexit
import inspect
#import numpy as np

import netCDF4
from readIODA2Obs import ReadIODA2Obs

#-------------------------------------------------------------------------------------
ERR_FILE = open('log.err', 'w+', encoding='utf-8')
LOG_FILE = open('log.out', 'w+', encoding='utf-8')

#-------------------------------------------------------------------------------------
def exit_handler():
 #ctrl + C works as well
 #log("Exiting")
  ERR_FILE.close()
  LOG_FILE.close()

# close files before exit
atexit.register(exit_handler)

#-------------------------------------------------------------------------------------
def log(*args, files=[sys.stdout, LOG_FILE]):
 # can also add timestamps etc.
  cf = inspect.currentframe()
  for f in files:
    print("DEBUG", f"{inspect.stack()[1][1]}:{cf.f_back.f_lineno}", *args, file=f)
    f.flush()

#-------------------------------------------------------------------------------------
def log_err(*args, files=[ERR_FILE, sys.stderr]):
  cf = inspect.currentframe()
  for f in files:
    print("ERROR", f"{inspect.stack()[1][1]}:{cf.f_back.f_lineno}", *args, file=f)
    f.flush()

#-------------------------------------------------------------------------------------
#log("log.out")
#log_err("log.err")

#-------------------------------------------------------------------------------------
def get_akbk(akbkfile):
  if(not os.path.exists(akbkfile)):
    print('File %s does not exist. Stop' %(abfkfile))
    sys.exit(-1)

  ncakbk = netCDF4.Dataset(akbkfile)
  ak = ncakbk.variables['ak'][:]
  bk = ncakbk.variables['bk'][:]
  ncakbk.close()

 #print('ak', ak)
 #print('bk', bk)
 #print('type(ak)', type(ak))
 #print('type(bk)', type(bk))

 #ps = 100.0*1050.0
 #print('ps', ps)
 #print('type(ps)', type(ps))

 #prs = ak + bk*ps

 #print('prs.shape', prs.shape)
 #for n in range(len(prs)):
 #  print('lev %d: ak %f bk %f prs %f' %(n, ak[n], bk[n], prs[n]))

  return ak, bk

#-------------------------------------------------------------------------------------
def get_grid_data(gridbase):
  if(not os.path.exists(gridbase)):
    print('File %s does not exist. Stop' %(gridbase))
    sys.exit(-1)

  ncgrid = netCDF4.Dataset(gridbase)
  lat = ncgrid.variables['lat'][:]
  lon = ncgrid.variables['lon'][:]
  psf = ncgrid.variables['ps'][0,:,:]
  tmp = ncgrid.variables['T'][0,:,:,:]

  print('lat.shape', lat.shape)
  print('lon.shape', lon.shape)
  print('psf.shape', psf.shape)
  print('tmp.shape', tmp.shape)

 #print('lat', lat)
 #print('lon', lon)
 #print('psf', psf[::5,::5])
 #print('tmp', tmp)

  ncgrid.close()

  return lat, lon, psf, tmp

#-------------------------------------------------------------------------------------
def get_obs_data(obsfile, debug=0):
  if(not os.path.exists(obsfile)):
    print('File %s does not exist. Stop' %(obsfile))
    sys.exit(-1)

  ncobs = ReadIODA2Obs(debug=debug, filename=obsfile)
  rawlat, rawlon = ncobs.get_latlon()
  varname = '/MetaData/pressure'
  rawprs = ncobs.get_var(varname)
 #varname = '/ObsError/airTemperature'
 #varname = '/GsiUseFlag/airTemperature'
 #varname = '/PreQC/airTemperature'
  varname = '/ObsValue/airTemperature'
  rawval = ncobs.get_var(varname)
  ncobs.close()

  obslon = []
  obslat = []
  obsprs = []
  obsval = []

  for n in range(len(rawlat)):
   #if(np.isnan(rawval[n])):
    if(math.isnan(rawval[n])):
      continue
    obslon.append(rawlon[n])
    obslat.append(rawlat[n])
    obsprs.append(rawprs[n])
    obsval.append(rawval[n])

 #for n in range(len(obslat)):
 #  print('No %d: lat %f lon %f prs %f tmp %f' %(n, obslon[n], obslat[n], obsprs[n], obsval[n]))

  return obslon, obslat, obsprs, obsval

