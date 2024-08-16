import getopt
import os, sys
import math
import numpy as np

import utils

from netCDF4 import Dataset
from plotUtils import PlotResult
from weightHandler import WeightReader
from weightHandler import DiagnosisWriter

#==================================================================================
class CheckNeuralNetwork(object):
  def __init__(self, debug=1, output=0,
               weightfile='crtm_weight.nc',
               diagnosisfile='diagnosis.nc',
               gridfile=None,
               obsfile=None):
   #parameters
    self.debug = debug
    self.output = output

    if(os.path.exists(gridfile)):
      print('background file %s' %(gridfile))
    else:
      print('background file %s does not exist. Exit.' %(gridfile))
      sys.exit(-1)
    if(os.path.exists(obsfile)):
      print('observation file %s' %(obsfile))
    else:
      print('observation file %s does not exist. Exit.' %(obsfile))
      sys.exit(-1)
    
    wf = Dataset(weightfile, 'r')
    self.bias1 = wf.variables['layer1bias'][:,:,:]
    self.weight1  = wf.variables['layer1wgt'][:,:,:,:]

    self.bias2 = wf.variables['layer2bias'][:,:,:]
    self.weight2  = wf.variables['layer2wgt'][:,:,:,:]
    wf.close()

    df = Dataset(diagnosisfile, 'r')
    self.xa = df.variables['diagnosis'][:,:,:]
    self.bt  = df.variables['brightnessTemperature'][:,:,:]
    self.hofx = df.variables['hofx'][:,:]
    df.close()

    self.layer2depth, self.nlat, self.nlon = self.bt.shape
    self.nlev, self.nlat, self.nlon = self.xa.shape
    self.nobs, self.nchn = self.hofx.shape

   #print('bias1.shape = ', self.bias1.shape)
   #print('weight1.shape = ', self.weight1.shape)
    print('bias2.shape = ', self.bias2.shape)
    print('weight2.shape = ', self.weight2.shape)
    print('self.nlon = ', self.nlon)
    print('self.nlat = ', self.nlat)
    print('self.nlev = ', self.nlev)
   #print('self.layer1depth = ', self.layer1depth)
    print('self.layer2depth = ', self.layer2depth)

    self.lat, self.lon, self.xb = utils.get_grid_data(gridfile)
    self.obslon, self.obslat, self.obsval, self.gsihofx, self.obs_qc = utils.get_obs_data(obsfile)

    print('self.xa.shape = ', self.xa.shape)
    print('self.hofx.shape = ', self.hofx.shape)

    self.pr = PlotResult(output=self.output)
    self.pr.set_obs_lonlat(self.obslon, self.obslat)

    for nbt in range(self.layer2depth):
      lvl = 8*nbt + 5
      title = 'Levev %d, Channel %d' %(lvl, nbt)
      imagename = 'lev%d_chn%d.png' %(lvl,nbt)
      self.pr.set_title(title)
      self.pr.set_imagename(imagename)

      self.obsdiff = self.obsval[:,nbt] - self.hofx[:,nbt]

      ba = self.xb[lvl,:,:]
      an = self.xa[lvl,:,:]
      dv = an - ba
      dv2 = self.xa[lvl,:,:] - self.xb[lvl,:,:]
      bt = self.bt[nbt,:,:]

      print('lvl %d: ba min: %f, max: %f' %(lvl, np.min(ba), np.max(ba)))
      print('lvl %d: an min: %f, max: %f' %(lvl, np.min(an), np.max(an)))
      print('lvl %d: dv min: %f, max: %f' %(lvl, np.min(dv), np.max(dv)))
      print('lvl %d: bt min: %f, max: %f' %(lvl, np.min(bt), np.max(bt)))
      print('omb min: %f, max: %f' %(np.min(self.obsdiff), np.max(self.obsdiff)))

      name = ['Background', 'BrightnessTemperature',
              'Analysis-Background', 'Anl-Bkg+ObsDiff']
      self.pr.set_runname(name)
      
      data = [ba, bt, dv, dv2]
      self.pr.plot(self.lon, self.lat, data=data, obsvar=self.obsdiff)
    
    self.pr.biasplot(self.bias1, 'Layer 1 Bias', 'layer1bias')
        
    self.pr.weightplot(self.weight1, title='Layer 1 Weight', imgname='layer1weight')
    
    self.pr.biasplot2(self.bias2, self.lon, self.lat, self.obslat, self.obslon,
                      self.obs_qc, title='Layer 2 Bias', imgname='layer2bias')
        
    self.pr.weightplot2(self.weight2, self.lon, self.lat, self.obslat, self.obslon,
                        self.obs_qc, title='Layer 2 Weight', imgname='layer2weight')

#===================================================================================
if __name__== '__main__':
  debug = 1
  output = 0

  dirname = '/work2/noaa/da/weihuang/EMC_cycling/jedi-cycling'
  datestr = '2022011400'

  opts, args = getopt.getopt(sys.argv[1:], '', ['debug=', 'output=',
                          'dirname=', 'datestr='])
  for o, a in opts:
    if o in ['--debug']:
      debug = int(a)
    elif o in ['--output']:
      output = int(a)
    elif o in ['--dirname']:
      dirname = a
    elif o in ['--datestr']:
      datestr = a
    else:
      assert False, 'unhandled option'

  gridfile = '%s/%s/sanl_%s_fhr06_ensmean' %(dirname, datestr, datestr)
  obsfile = '%s/%s/ioda_v2_data/amsua_n19_obs_%s.nc4' %(dirname, datestr, datestr)
  weightfile = 'saved.weight/crtm_weight_%s.nc' %(datestr)
  diagfile = 'saved.weight/diagnosis_%s.nc' %(datestr)

  cnn = CheckNeuralNetwork(debug=debug, output=output,
                           weightfile=weightfile,
                           diagnosisfile=diagfile,
                           gridfile=gridfile,
                           obsfile=obsfile)

