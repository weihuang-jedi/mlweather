import os, sys
import math
import atexit
import getopt
import inspect
import numpy as np

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
def get_grid_data(filename):
  if(not os.path.exists(filename)):
    print('File %s does not exist. Stop' %(filename))
    sys.exit(-1)

  ncgrid = netCDF4.Dataset(filename)
  lat = ncgrid.variables['lat'][:,:]
  lon = ncgrid.variables['lon'][:,:]
  tmp = ncgrid.variables['tmp'][0,:,:,:]

  print('lat.shape', lat.shape)
  print('lon.shape', lon.shape)
  print('tmp.shape', tmp.shape)

 #print('lat', lat)
 #print('lon', lon)
 #print('tmp', tmp)

  ncgrid.close()

  return lat[:,0], lon[0,:], tmp

#-------------------------------------------------------------------------------------
def get_obs_data(obsfile, debug=0, channel=0):
  if(not os.path.exists(obsfile)):
    print('File %s does not exist. Stop' %(obsfile))
    sys.exit(-1)

  ncobs = ReadIODA2Obs(debug=debug, filename=obsfile)
  rawlat, rawlon = ncobs.get_latlon()
 #varname = '/MetaData/pressure'
 #varname = '/MetaData/height'
 #rawprs = ncobs.get_var(varname)
  varname = '/ObsValue/brightnessTemperature'
  rawval = ncobs.get_2d_var(varname)
 #varname = '/GsiHofX/brightnessTemperature'
  varname = '/GsiHofXBc/brightnessTemperature'
  rawgsi = ncobs.get_2d_var(varname)
  varname = '/PreQC/brightnessTemperature'
  raw_qc = ncobs.get_2d_var(varname)
  ncobs.close()

  nlocs, nchannels = rawval.shape

  idxlist = []

  for n in range(nlocs):
   #if(np.isnan(rawval[n,channel])):
    if(math.isnan(rawval[n,channel])):
      continue
    idxlist.append(n)

  nv = len(idxlist)
  obslon = np.ndarray((nv, ), dtype=float)
  obslat = np.ndarray((nv, ), dtype=float)
  obsval = np.ndarray((nv, nchannels), dtype=float)
  gsihox = np.ndarray((nv, nchannels), dtype=float)
  obs_qc = np.ndarray((nv, nchannels), dtype=float)
  
  for i in range(nv):
    n = idxlist[i]
    obslon[i] = rawlon[n]
    obslat[i] = rawlat[n]
    obsval[i,:] = rawval[n,:]
    gsihox[i,:] = gsihox[n,:]
    obs_qc[i,:] = raw_qc[n,:]

 #for n in range(len(obslat)):
 #  print('No %d: lat %f lon %f tmp %f' %(n, obslon[n], obslat[n], obsval[n,0]))

  return obslon, obslat, obsval, gsihox, obs_qc

#===================================================================================
if __name__== '__main__':
  debug = 1
  dirname = '/work2/noaa/da/weihuang/EMC_cycling/jedi-cycling'
  datestr = '2022010400'

  opts, args = getopt.getopt(sys.argv[1:], '', ['debug=', 'dirname=', 'datestr='])
  for o, a in opts:
    if o in ('--debug'):
      debug = int(a)
    elif o in ('--dirname'):
      dirname = a
    elif o in ('--datestr'):
      datestr = a
    else:
      assert False, 'unhandled option'

  filename = '%s/%s/sanl_%s_fhr06_ensmean' %(dirname, datestr, datestr)

  lat, lon, tmp = get_grid_data(filename)

  nlev, nlat, nlon = tmp.shape

 #for j in range(0, nlat, 20):
 #  for i in range(0, nlon, 20):
 #    print('lon %f, lat %f, t %f' %(lon[j,i], lat[j,i], tmp[nlev-1,j,i]))

  for j in range(0, nlat, 20):
    print('lat %f, t %f' %(lat[j], tmp[nlev-1,j,0]))

  for i in range(0, nlon, 20):
    print('lon %f, t %f' %(lon[i], tmp[nlev-1,0,i]))

