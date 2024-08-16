import getopt
import os, sys
import math
import numpy as np

import nnutils
import utils

from plotUtils import PlotResult
from weightHandler import WeightReader
from weightHandler import WeightWriter
from weightHandler import DiagnosisWriter

#==================================================================================
class NeuralNetwork(object):
  def __init__(self, lon, lat, xb, obslon, obslat,
               obsval, gsihofx, obs_qc, debug=1):
   #parameters
    self.debug = 1
    self.prev_cost = 1.0e+21
    self.cost = 1.0e+20
   #self.step = 1.0
   #self.step = 0.625
    self.step = 0.5
   #self.step = 0.1

    self.lon = lon
    self.lat = lat
    self.xa = np.copy(xb)
    self.xb = xb
    self.obslon = obslon
    self.obslat = obslat
    self.obsval = obsval
    self.obs_qc = obs_qc
    self.gsihofx = gsihofx

    self.obsanl = np.copy(obsval)

    self.nlev, self.nlat, self.nlon = xb.shape
    self.nwidth = 3
    self.ndepth = 3
    self.layer1depth = self.nlev
    self.nobs, self.layer2depth = obsval.shape
    self.nchn = self.layer2depth

    print('nlon: %d, nlat: %d, nlev: %d' %(self.nlon, self.nlat, self.nlev))
    print('nobs: %d, layer2depth: %d' %(self.nobs, self.layer2depth))

    print('lon.shape = ', lon.shape)
    print('lat.shape = ', lat.shape)
    print('xb.shape = ', xb.shape)
    print('obslon.shape = ', obslon.shape)
    print('obslat.shape = ', obslat.shape)
    print('obsval.shape = ', obsval.shape)

    self.weightreader = WeightReader(debug=self.debug)
    self.weightwriter = WeightWriter(debug=self.debug)
    self.diagnosiswriter = DiagnosisWriter(debug=self.debug)

   #print(nnutils.initialize.__doc__)
    rc = nnutils.initialize(lon, lat, xb, obslon, obslat, obsval, obs_qc,
                            self.nlon, self.nlat, self.nlev,
                            self.nobs, self.layer2depth)
    rc = nnutils.initialize_layer1(self.layer1depth)
    rc = nnutils.initialize_layer2(self.layer2depth)

    xbmin = np.min(xb)
    xbmax = np.max(xb)
    scalefactor = xbmax-xbmin
    offset = xbmin + 0.5*scalefactor
   #print(nnutils.preconditioning.__doc__)
    rc = nnutils.preconditioning(offset, scalefactor)

#-------------------------------------------------------------------------------------------
  def finalize(self):
   #print(nnutils.finalize.__doc__)
    rc = nnutils.finalize_layer2()
    rc = nnutils.finalize_layer1()
    rc = nnutils.finalize()

#-------------------------------------------------------------------------------------------
  def increase_step(self):
    self.step *= 1.125
        
#-------------------------------------------------------------------------------------------
  def decrease_step(self):
    self.step *= 0.75
        
#-------------------------------------------------------------------------------------------
  def update(self):
   #update state
    rc = nnutils.update()

#-------------------------------------------------------------------------------------------
  def forward(self):
   #if(self.debug):
   #  utils.log('forward')
   #print(nnutils.__doc__)
   #print(nnutils.forward.__doc__)

   #forward propagation
    self.cost = nnutils.forward()

   #print('in forward, self.cost = ', self.cost)
        
#-------------------------------------------------------------------------------------------
  def get_xaya(self):
   #if(self.debug):
   #  utils.log('get_xaya')
   #print(nnutils.get_analysis.__doc__)
    return nnutils.get_analysis(self.nlon, self.nlat, self.nlev, self.nobs, self.layer2depth)

#-------------------------------------------------------------------------------------------
  def get_bt(self):
   #if(self.debug):
   #  utils.log('get_bt')
   #print(nnutils.get_layer2_var.__doc__)
    return nnutils.get_layer2_var(self.nlon, self.nlat, self.layer2depth)

#-------------------------------------------------------------------------------------------
  def get_precost(self):
    return self.prev_cost

#-------------------------------------------------------------------------------------------
  def get_curcost(self):
    return self.cost

#-------------------------------------------------------------------------------------------
  def backward(self):
   #if(self.debug):
   #  utils.log('backward')

   #print(nnutils.__doc__)
   #print(nnutils.backward.__doc__)

   #backward propogate through the network
    rc = nnutils.backward(self.step)
        
#-------------------------------------------------------------------------------------------
  def test_train(self, step):
    if(self.debug):
      utils.log('pre_train')

    self.step = step

    self.forward()

    self.xa, self.obsanl = self.get_xaya()

    pinfo = 'test_train precost: %e, ' %(self.prev_cost)
    pinfo = '%s curcost: %e' %(pinfo, self.cost)
    print(pinfo)

    self.backward()

    self.forward()
   #rc = nnutils.update()

#-------------------------------------------------------------------------------------------
  def pre_train(self, step):
   #if(self.debug):
   #  utils.log('pre_train')

    self.step = step
    return_status = True
    maxpretrain = 30
    numpretrain = 0
    while(numpretrain < maxpretrain):
      numpretrain += 1
      self.forward()

      pinfo = 'Number pre-train %d: ' %(numpretrain)
      pinfo = '%s precost: %e, ' %(pinfo, self.prev_cost)
      pinfo = '%s curcost: %e' %(pinfo, self.cost)
      print(pinfo)

      if(self.cost >= self.prev_cost):
        numpretrain += maxpretrain
        return_status = False

      self.prev_cost = self.cost

      self.backward()

    return return_status

#-------------------------------------------------------------------------------------------
  def save_weightNbias(self, filename='crtm_weight.nc'):
    if(os.path.exists(filename)):
      cmd = 'rm -f old-%s.nc' %(filename)
      os.system(cmd)
      cmd = 'mv %s old-%s.nc' %(filename, filename)
      os.system(cmd)
    self.weightwriter.createFile(filename=filename)
    self.weightwriter.createDimension(self.nlon, self.nlat, self.nlev,
                                       self.nwidth, self.ndepth,
                                       self.layer1depth, self.layer2depth)
    self.weightwriter.createVariable()
    self.weightwriter.writeDimension()

    self.bias1, self.weight1 = nnutils.get_layer1_weight(self.nlon, self.nlat,
                               self.nlev, self.layer1depth)
    self.bias2, self.weight2 = nnutils.get_layer2_weight(self.nlon, self.nlat,
                               self.layer1depth, self.layer2depth)

    self.weightwriter.writeLayer2BiasWeight(self.bias2, self.weight2)
    self.weightwriter.writeLayer1BiasWeight(self.bias1, self.weight1)
    self.weightwriter.closeFile()

    print('bias1 min: %e, max: %e' %(np.min(self.bias1), np.max(self.bias1)))
   #print('bias1 = ', self.bias1)
    print('bias2 min: %e, max: %e' %(np.min(self.bias2), np.max(self.bias2)))
   #print('bias2 = ', self.bias2)
    print('weight1 min: %e, max: %e' %(np.min(self.weight1), np.max(self.weight1)))
   #print('weight1 = ', self.weight1)
    print('weight2 min: %e, max: %e' %(np.min(self.weight2), np.max(self.weight2)))
   #print('weight2 = ', self.weight2)

    print('nan indices in weight1:', np.argwhere(np.isnan(self.weight1)))

#-------------------------------------------------------------------------------------------
  def saveDiagnosis(self, filename='diagnosis.nc'):
    if(os.path.exists(filename)):
      cmd = 'rm -f old-%s.nc' %(filename)
      os.system(cmd)
      cmd = 'mv %s old-%s.nc' %(filename, filename)
      os.system(cmd)
    self.diagnosiswriter.createFile(filename=filename)
    self.diagnosiswriter.createDimension(self.nlon, self.nlat, self.nlev,
                                         self.nobs, self.nchn)
    self.diagnosiswriter.createVariable()
    self.diagnosiswriter.writeDimension()

    self.bt = self.get_bt()
    self.xa, self.obsanl = self.get_xaya()
    self.diagnosiswriter.writeDiagnosis(self.xa, self.bt, self.obsanl)
    self.diagnosiswriter.closeFile()

    print('analysis min: %e, max: %e' %(np.min(self.xa), np.max(self.xa)))
    print('brightnessTemperature min: %e, max: %e' %(np.min(self.bt), np.max(self.bt)))
    print('hofx min: %e, max: %e' %(np.min(self.obsanl), np.max(self.obsanl)))

#-------------------------------------------------------------------------------------------
  def useSavedWeight(self, weightfilename='old-crtm_weight.nc'):
   #print(nnutils.__doc__)
   #print(nnutils.set_layer1_weight.__doc__)
   #print(nnutils.set_layer2_weight.__doc__)

    bias1, weight1, bias2, weight2 = self.weightreader.readWeightFile(weightfilename=weightfilename)

    print('bias1.shape = ', bias1.shape)
    print('weight1.shape = ', weight1.shape)
    print('bias2.shape = ', bias2.shape)
    print('weight2.shape = ', weight2.shape)
    print('self.nlon = ', self.nlon)
    print('self.nlat = ', self.nlat)
    print('self.nlev = ', self.nlev)
    print('self.layer1depth = ', self.layer1depth)
    print('self.layer2depth = ', self.layer2depth)

    rc = nnutils.set_layer1_weight(bias1, weight1, self.nlon, self.nlat,
                                   self.nlev, self.layer1depth)
    rc = nnutils.set_layer2_weight(bias2, weight2, self.nlon, self.nlat,
                                   self.layer1depth, self.layer2depth)
    print('Done set layer weight')

#-------------------------------------------------------------------------------------------
  def set_prev_cost(self, cost):
    self.prev_cost = cost

#-------------------------------------------------------------------------------------------
  def train(self, step):
   #if(self.debug):
   #  utils.log('train')

    self.step = step

    self.forward()

   #print('Precost: %e, Curcost: %e' %(self.prev_cost, self.cost))

    if(self.cost >= self.prev_cost):
      return True

    self.prev_cost = self.cost

    self.backward()

    return False

#------------------------------------------------------------------------
  def view_status(self, pr, title='Status'):
    self.xa, self.obsanl = self.get_xaya()
    obsdiff = self.obsval[:,0] - self.obsanl[:,0]

   #print('self.xa.shape = ', self.xa.shape)
   #print('self.obsanl.shape = ', self.obsanl.shape)

    bt2 = self.get_bt()
   #for lvl in range(50, self.nlev, 10):
    for lvl in range(50, 100, 20):
      newtitle = '%s lev %d' %(title, lvl)
      imagename = 'status_lev%d.png' %(lvl)
      pr.set_title(newtitle)
      pr.set_imagename(imagename)
      ba = self.xb[lvl,:,:]
      an = self.xa[lvl,:,:]
      bt = bt2[0,:,:]
      dv = an - ba

      print('lvl %d: ba min: %f, max: %f' %(lvl, np.min(ba), np.max(ba)))
      print('lvl %d: an min: %f, max: %f' %(lvl, np.min(an), np.max(an)))
      print('lvl %d: dv min: %f, max: %f' %(lvl, np.min(dv), np.max(dv)))
      print('lvl %d: bt min: %f, max: %f' %(lvl, np.min(bt), np.max(bt)))
      print('lvl %d:omb min: %f, max: %f' %(lvl, np.min(obsdiff), np.max(obsdiff)))
      data = [ba, an, bt, dv]
      pr.plot(self.lon, self.lat, data=data, obsvar=obsdiff)

#------------------------------------------------------------------------
  def plot_result(self, pr, iteration=-1):
    self.xa, self.obsanl = self.get_xaya()
    obsdiff = self.obsval[:,0] - self.obsanl[:,0]

    bt2 = self.get_bt()
   #for lvl in range(50, self.nlev, 10):
    for lvl in range(50, 100, 20):
      if(iteration < 0):
        title = 'Final No Smooth lev %d' %(lvl)
        imagename = 'final-no-smooth_lev%d.png' %(lvl)
      else:
        title = 'No Smooth Iteration %d lev %d' %(iteration, lvl)
        imagename = 'no-smooth-iteration-%d_lev%d.png' %(iteration, lvl)
      pr.set_title(title)
      pr.set_imagename(imagename)

      ba = self.xb[lvl,:,:]
      an = self.xa[lvl,:,:]
      dv = an - ba
      bt = bt2[0,:,:]

      print('lvl %d: ba min: %f, max: %f' %(lvl, np.min(ba), np.max(ba)))
      print('lvl %d: an min: %f, max: %f' %(lvl, np.min(an), np.max(an)))
      print('lvl %d: dv min: %f, max: %f' %(lvl, np.min(dv), np.max(dv)))
      print('lvl %d: bt min: %f, max: %f' %(lvl, np.min(bt), np.max(bt)))
      print('lvl %d:omb min: %f, max: %f' %(lvl, np.min(obsdiff), np.max(obsdiff)))
      data = [ba, an, bt, dv]
      pr.plot(self.lon, self.lat, data=data, obsvar=obsdiff)
    
   #------------------------------------------------------------------------
    self.xa = nnutils.light_smooth(self.nlon, self.nlat, self.nlev)

    bt2 = self.get_bt()
   #for lvl in range(50, self.nlev, 10):
    for lvl in range(50, 100, 20):
      if(iteration < 0):
        title = 'Final Light Smooth lev %d' %(lvl)
        imagename = 'final-light-smooth_lev%d.png' %(lvl)
      else:
        title = 'Light Smooth Iteration %d lev %d' %(iteration, lvl)
        imagename = 'light-smooth-iteration-%d_lev%d.png' %(iteration, lvl)
      pr.set_title(title)
      pr.set_imagename(imagename)

      ba = self.xb[lvl,:,:]
      an = self.xa[lvl,:,:]
      dv = an - ba
      bt = bt2[0,:,:]

      print('lvl %d: ba min: %f, max: %f' %(lvl, np.min(ba), np.max(ba)))
      print('lvl %d: an min: %f, max: %f' %(lvl, np.min(an), np.max(an)))
      print('lvl %d: dv min: %f, max: %f' %(lvl, np.min(dv), np.max(dv)))
      print('lvl %d: bt min: %f, max: %f' %(lvl, np.min(bt), np.max(bt)))
      print('lvl %d:omb min: %f, max: %f' %(lvl, np.min(obsdiff), np.max(obsdiff)))
      data = [ba, an, bt, dv]
      pr.plot(self.lon, self.lat, data=data, obsvar=obsdiff)
    
   #------------------------------------------------------------------------
    self.xa = nnutils.heavy_smooth(self.nlon, self.nlat, self.nlev)
    
    bt = self.get_bt()
   #for lvl in range(50, self.nlev, 10):
    for lvl in range(50, 100, 20):
      if(iteration < 0):
        title = 'Final Heavy Smooth lev %d' %(lvl)
        imagename = 'final-heavy-smooth_lev%d.png' %(lvl)
      else: 
        title = 'Heavy Smooth Iteration %d lev %d' %(iteration, lvl)
        imagename = 'heavy-smooth-iteration-%d_lev%d.png' %(iteration, lvl)
      pr.set_title(title)
      pr.set_imagename(imagename)

      ba = self.xb[lvl,:,:]
      an = self.xa[lvl,:,:]
      bt = bt2[0,:,:]
      dv = an - ba

      print('lvl %d: ba min: %f, max: %f' %(lvl, np.min(ba), np.max(ba)))
      print('lvl %d: an min: %f, max: %f' %(lvl, np.min(an), np.max(an)))
      print('lvl %d: dv min: %f, max: %f' %(lvl, np.min(dv), np.max(dv)))
      print('lvl %d: bt min: %f, max: %f' %(lvl, np.min(bt), np.max(bt)))
      print('lvl %d:omb min: %f, max: %f' %(lvl, np.min(obsdiff), np.max(obsdiff)))
      data = [ba, an, bt, dv]
      pr.plot(self.lon, self.lat, data=data, obsvar=obsdiff)

#------------------------------------------------------------------------
  def plot_bias2(self, pr):
   #bias1_cnt, weight1_cnt = nnutils.get_layer1_count(self.nlon, self.nlat,
   #                                                  self.nlev, self.layer1depth)
   #pr.lineplot(bias1, bias1_cnt, title='Layer 1 Bias', imgname='layer1bias')

    bias2_cnt, weight2_cnt = nnutils.get_layer2_count(self.nlon, self.nlat,
                                                      self.layer1depth, self.layer2depth)
    pr.biasplot(self.bias2, bias2_cnt, title='Layer 2 Bias', imgname='layer2bias')
        
#------------------------------------------------------------------------
  def plot_weight2(self, pr):
   #bias1_cnt, weight1_cnt = nnutils.get_layer1_count(self.nlon, self.nlat,
   #                                                  self.nlev, self.layer1depth)
   #pr.lineplot(bias1, bias1_cnt, title='Layer 1 Bias', imgname='layer1bias')

    bias2_cnt, weight2_cnt = nnutils.get_layer2_count(self.nlon, self.nlat,
                                                      self.layer1depth, self.layer2depth)
    pr.weightplot(self.weight2, bias2_cnt, title='Layer 2 Bias', imgname='layer2weight')

#===================================================================================
