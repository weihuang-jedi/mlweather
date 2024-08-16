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
  lat = ncgrid.variables['lat_0'][:]
  lon = ncgrid.variables['lon_0'][:]
  prs = ncgrid.variables['lv_ISBL0'][:]
  tmp = ncgrid.variables['TMP_P0_L100_GLL0'][:,:,:]

  print('lat.shape', lat.shape)
  print('lon.shape', lon.shape)
  print('tmp.shape', tmp.shape)

 #print('lat', lat)
 #print('lon', lon)
 #print('tmp', tmp)

  ncgrid.close()

  return lat, lon, prs, tmp

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

  lat, lon, prs, tmp = get_grid_data(filename)

  nlev, nlat, nprs, nlon = tmp.shape

  for i in range(0, nlon, 20):
    print('#%d: lon %f, t %f' %(i, lon[i]))

  for j in range(0, nlat, 20):
    print('#%d: lat %f, t %f' %(j, lat[j]))

  for k in range(nprs):
    print('Prs %d: %f' %(k, prs[k-1]))

