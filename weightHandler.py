from netCDF4 import Dataset
import numpy as np
import getopt
import os, sys

#=========================================================================
class WeightReader():
  def __init__(self, debug=0):
    self.debug = debug

    if(self.debug):
      print('debug = ', debug)

    self.set_defaults()

#------------------------------------------------------------------
  def set_defaults(self):
    self.name = 'WeightReader'

#------------------------------------------------------------------
  def readWeightFile(self, weightfilename='old-crtm_weight.nc'):
    wf = Dataset(weightfilename, 'r')
    layer1bias = wf.variables['layer1bias'][:,:,:]
    layer1wgt  = wf.variables['layer1wgt'][:,:,:,:]
    layer2bias = wf.variables['layer2bias'][:,:,:]
    layer2wgt  = wf.variables['layer2wgt'][:,:,:,:]
    wf.close()
    return layer1bias, layer1wgt, layer2bias, layer2wgt

#=========================================================================
class WeightWriter():
  def __init__(self, debug=0):
    self.debug = debug

    if(self.debug):
      print('debug = ', debug)

    self.set_defaults()

#------------------------------------------------------------------
  def set_defaults(self):
    self.name = 'WeightWriter'

#------------------------------------------------------------------
  def createFile(self, filename):
    if(os.path.exists(filename)):
      cmd = 'mv %s old-%s.nc' %(filename, filename)
      os.system(cmd)
    self.ncfile = Dataset(filename, 'w', format='NETCDF4')
    self.ncfile.description = 'CRMT Neural Network Weights'

#------------------------------------------------------------------
  def closeFile(self):
    self.ncfile.close()

#------------------------------------------------------------------
  def createDimension(self, nlon, nlat, nlev, nwidth, ndepth,
                      nlayer1depth, nlayer2depth):
    self.nlon = nlon
    self.nlat = nlat
    self.nlev = nlev
    self.nwidth = nwidth
    self.ndepth = ndepth
    self.nlayer1depth = nlayer1depth
    self.nlayer2depth = nlayer2depth

   #dimensions
    self.ncfile.createDimension('lon', nlon)
    self.ncfile.createDimension('lat', nlat)
    self.ncfile.createDimension('lev', nlev)
    self.ncfile.createDimension('xwidth', nwidth)
    self.ncfile.createDimension('ywidth', nwidth)
    self.ncfile.createDimension('zdepth', ndepth)
    self.ncfile.createDimension('layer1depth', nlayer1depth)
    self.ncfile.createDimension('layer2depth', nlayer2depth)

#------------------------------------------------------------------
  def createVariable(self):
   #dimension variables
    self.lon = self.ncfile.createVariable('lon', 'f4', ('lon',))
    self.lon.units = 'degrees_east'
    self.lon.long_name = 'longitude'

    self.lat = self.ncfile.createVariable('lat', 'f4', ('lat',))
    self.lat.units = 'degrees_north'
    self.lat.long_name = 'latitude'

    self.lev = self.ncfile.createVariable('lev', 'int32', ('lev',))
    self.lev.units = 'top_down'
    self.lev.long_name = 'level'

    self.xwidth = self.ncfile.createVariable('xwidth', 'int32', ('xwidth',))
    self.xwidth.units = 'none'
    self.xwidth.long_name = 'width in x-direction'

    self.ywidth = self.ncfile.createVariable('ywidth', 'int32', ('ywidth',))
    self.ywidth.units = 'none'
    self.ywidth.long_name = 'width in y-direction'

    self.zdepth = self.ncfile.createVariable('zdepth', 'int32', ('zdepth',))
    self.zdepth.units = 'none'
    self.zdepth.long_name = 'depth in z-direction'

    self.layer1depth = self.ncfile.createVariable('layer1depth', 'int32', ('layer1depth',))
    self.layer1depth.units = 'none'
    self.layer1depth.long_name = 'number of layer1'

    self.layer2depth = self.ncfile.createVariable('layer2depth', 'int32', ('layer2depth',))
    self.layer2depth.units = 'none'
    self.layer2depth.long_name = 'number of layer2'

   #variables
   #self.grdbias = self.ncfile.createVariable('grdbias', 'f4', ('lev', 'lat', 'lon'))
   #self.grdbias.units = 'none'
   #self.grdbias.long_name = 'Grid Bias'

   #self.grdwgt  = self.ncfile.createVariable('grdwgt', 'f4', ('zdepth', 'ywidth', 'xwidth', 'lev', 'lat', 'lon'))
   #self.grdwgt.units = 'none'
   #self.grdwgt.long_name = 'Grid Weight'

   #self.analysis = self.ncfile.createVariable('analysis', 'f4', ('lev', 'lat', 'lon'))
   #self.analysis.units = 'K'
   #self.analysis.long_name = 'Temperature'

    self.layer1bias = self.ncfile.createVariable('layer1bias', 'f4', ('layer1depth', 'lat', 'lon'))
    self.layer1bias.units = 'none'
    self.layer1bias.long_name = 'Layer 1 Bias'
   #self.layer1bias._FillValue = 9.96921e+36

    self.layer1wgt  = self.ncfile.createVariable('layer1wgt', 'f4', ('layer1depth', 'lev', 'lat', 'lon'))
    self.layer1wgt.units = 'none'
    self.layer1wgt.long_name = 'Layer 1 Weight'
   #self.layer1wgt._FillValue = 9.96921e+36

    self.layer2bias = self.ncfile.createVariable('layer2bias', 'f4', ('layer2depth', 'lat', 'lon'))
    self.layer2bias.units = 'none'
    self.layer2bias.long_name = 'Layer 2 Bias'
   #self.layer2bias._FillValue = 9.96921e+36

    self.layer2wgt  = self.ncfile.createVariable('layer2wgt', 'f4', ('layer2depth', 'layer1depth', 'lat', 'lon'))
    self.layer2wgt.units = 'none'
    self.layer2wgt.long_name = 'Layer 2 Weight'
   #self.layer2wgt._FillValue = 9.96921e+36

#------------------------------------------------------------------
  def writeDimension(self):
   #dimension value
    dlon = 360.0/self.nlon
    dlat = 180.0/(self.nlat-1)
    vlon = np.linspace(0, 360-dlon, self.nlon)
    vlat = np.linspace(0, 180, self.nlat)
    vlat = vlat - 90.0
    vlev = np.linspace(0, self.nlev-1, self.nlev)
    vxwidth = np.linspace(0, self.nwidth-1, self.nwidth)
    vywidth = np.linspace(0, self.nwidth-1, self.nwidth)
    vzdepth = np.linspace(0, self.ndepth-1, self.ndepth)
    vlayer1depth = np.linspace(0, self.nlayer1depth-1, self.nlayer1depth)
    vlayer2depth = np.linspace(0, self.nlayer2depth-1, self.nlayer2depth)

    self.lon[:] = vlon
    self.lat[:] = vlat
    self.lev[:] = vlev
    self.xwidth[:] = vxwidth
    self.ywidth[:] = vywidth
    self.zdepth[:] = vzdepth
    self.layer1depth[:] = vlayer1depth
    self.layer2depth[:] = vlayer2depth

   #print('lon=', vlon)
   #print('lat=', vlat)
   #print('lev=', vlev)
   #print('xwidth=', vxwidth)
   #print('ywidth=', vywidth)
   #print('zdepth=', vzdepth)
   #print('layer1depth=', vlayer1depth)
   #print('layer2depth=', vlayer2depth)

#------------------------------------------------------------------
 #def writeAnalysis(self, analysis):
 #  self.analysis[:,:,:] = analysis

#------------------------------------------------------------------
  def writeLayer1BiasWeight(self, bias, weight):
    self.layer1bias[:,:,:] = bias
    self.layer1wgt[:,:,:,:] = weight

#------------------------------------------------------------------
  def writeLayer2BiasWeight(self, bias, weight):
    self.layer2bias[:,:,:] = bias
    self.layer2wgt[:,:,:,:] = weight

#=========================================================================
class DiagnosisReader():
  def __init__(self, debug=0):
    self.debug = debug

    if(self.debug):
      print('debug = ', debug)

    self.set_defaults()

#------------------------------------------------------------------
  def set_defaults(self):
    self.name = 'DiagnosisReader'

#------------------------------------------------------------------
  def readDiagnosisFile(self, filename):
    if(not os.path.exists(filename)):
      print('Diagnosis file %s does not exist. Exit.' %(filename))
      sys.exit(-1)
    df = Dataset(filename, 'r')
    xa = df.variables['diagnosis'][:,:,:]
    bt  = df.variables['brightnessTemperature'][:,:,:]
    hofx = df.variables['hofx'][:,:,:]
    df.close()

    return xa, bt, hofx

#=========================================================================
class DiagnosisWriter():
  def __init__(self, debug=0):
    self.debug = debug

    if(self.debug):
      print('debug = ', debug)

    self.set_defaults()

#------------------------------------------------------------------
  def set_defaults(self):
    self.name = 'DiagnosisWriter'

#------------------------------------------------------------------
  def createFile(self, filename):
    if(os.path.exists(filename)):
      cmd = 'mv %s %s' %(filename, filename)
      os.system(cmd)
    self.ncfile = Dataset(filename, 'w', format='NETCDF4')
    self.ncfile.description = 'Diagnose Variables: xa, brightnessTemperature, and hofx'

#------------------------------------------------------------------
  def closeFile(self):
    self.ncfile.close()

#------------------------------------------------------------------
  def createDimension(self, nlon, nlat, nlev, nobs, nchn):
    self.nlon = nlon
    self.nlat = nlat
    self.nlev = nlev
    self.nobs = nobs
    self.nchn = nchn

   #dimensions
    self.ncfile.createDimension('lon', nlon)
    self.ncfile.createDimension('lat', nlat)
    self.ncfile.createDimension('lev', nlev)
    self.ncfile.createDimension('Location', nobs)
    self.ncfile.createDimension('Channel', nchn)

#------------------------------------------------------------------
  def createVariable(self):
   #dimension variables
    self.lon = self.ncfile.createVariable('lon', 'f4', ('lon',))
    self.lon.units = 'degrees_east'
    self.lon.long_name = 'longitude'

    self.lat = self.ncfile.createVariable('lat', 'f4', ('lat',))
    self.lat.units = 'degrees_north'
    self.lat.long_name = 'latitude'

    self.lev = self.ncfile.createVariable('lev', 'int32', ('lev',))
    self.lev.units = 'top_down'
    self.lev.long_name = 'level'

    self.location = self.ncfile.createVariable('Location', 'int32', ('Location',))
    self.location.units = 'none'
    self.location.long_name = 'Location'

    self.channel = self.ncfile.createVariable('Channel', 'int32', ('Channel',))
    self.channel.units = 'none'
    self.channel.long_name = 'Channel'

   #variables
    self.diagnosis = self.ncfile.createVariable('diagnosis', 'f4', ('lev', 'lat', 'lon'))
    self.diagnosis.units = 'K'
    self.diagnosis.long_name = 'Temperature'
   #self.diagnosis._FillValue = 9.96921e+36

    self.brightnessTemperature = self.ncfile.createVariable('brightnessTemperature', 'f4', ('Channel', 'lat', 'lon'))
    self.brightnessTemperature.units = 'none'
    self.brightnessTemperature.long_name = 'brightnessTemperature'
   #self.brightnessTemperature._FillValue = 9.96921e+36

    self.hofx  = self.ncfile.createVariable('hofx', 'f4', ('Location', 'Channel'))
    self.hofx.units = 'none'
    self.hofx.long_name = 'Observation - background brightnessTemperature'
   #self.hofx._FillValue = 9.96921e+36

#------------------------------------------------------------------
  def writeDimension(self):
   #dimension value
    dlon = 360.0/self.nlon
    dlat = 180.0/(self.nlat-1)
    vlon = np.linspace(0, 360-dlon, self.nlon)
    vlat = np.linspace(0, 180, self.nlat)
    vlat = vlat - 90.0
    vlev = np.linspace(0, self.nlev-1, self.nlev)
    vloc = np.linspace(0, self.nobs-1, self.nobs)
    vchn = np.linspace(0, self.nchn-1, self.nchn)

    self.lon[:] = vlon
    self.lat[:] = vlat
    self.lev[:] = vlev
    self.location[:] = vloc
    self.channel[:] = vchn

   #print('lon=', vlon)
   #print('lat=', vlat)
   #print('lev=', vlev)
   #print('loc=', vloc)
   #print('chn=', vchn)

#------------------------------------------------------------------
  def writeDiagnosis(self, diagnosis, bt, hofx):
    self.diagnosis[:,:,:] = diagnosis
    self.brightnessTemperature[:,:,:] = bt
    self.hofx[:,:] = hofx

#------------------------------------------------------------------
if __name__ == '__main__':
  debug = 1
  filename = 'my_netcdf.nc'

  opts, args = getopt.getopt(sys.argv[1:], '', ['debug=', 'filename='])

  for o, a in opts:
    if o in ('--debug'):
      debug = int(a)
    elif o in ('--filename'):
      filename = a
   #else:
   #  assert False, 'unhandled option'

 #print('debug = ', debug)
 #print('filename = ', filename)

  nlon = 48
  nlat = 25
  nlev = 11
  nwidth = 3
  ndepth = 3
  nlayer1depth = nlev
  nlayer2depth = 7

  wh = WeightHandler(debug=debug, filename=filename)
  wh.createDimension(nlon, nlat, nlev, nwidth, ndepth,
                     nlayer1depth, nlayer2depth)
  wh.createVariable()
  wh.writeDimension()
  wh.close()

