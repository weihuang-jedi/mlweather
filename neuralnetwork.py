import getopt
import os, sys
import math
import numpy as np

import nnutils
import utils

from plotUtils import PlotResult

#==================================================================================
class NeuralNetwork(object):
  def __init__(self, lon, lat, ak, bk, ps, xb,
               obslon, obslat, obsprs, obsval, debug=1):
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
    self.ak = ak
    self.bk = bk
    self.ps = ps

    self.nlon = len(lon)
    self.nlat = len(lat)
    self.npfl = len(ak)
    self.nprs = len(ak) - 1

    print('nlon: %d, nlat: %d, npfl: %d' %(self.nlon, self.nlat, self.npfl))

    self.xa = np.copy(xb)
    self.xb = xb

    self.obslon = obslon
    self.obslat = obslat
    self.obsprs = obsprs
    self.obsval = obsval

    self.nobs = len(obsval)

    self.obsanl = np.ndarray((self.nobs,), dtype=float)

   #print('lon.shape = ', lon.shape)
   #print('lat.shape = ', lat.shape)
   #print('ak.shape = ', ak.shape)
   #print('bk.shape = ', bk.shape)
   #print('ps.shape = ', ps.shape)
   #print('xb.shape = ', xb.shape)

   #print(nnutils.initialize.__doc__)
    rc = nnutils.initialize(lon, lat, ak, bk, ps, xb,
                            obslon, obslat, obsprs, obsval,
                            self.nlon, self.nlat, self.nprs, self.nobs)

    xbmin = np.min(xb)
    xbmax = np.max(xb)
    scalefactor = xbmax-xbmin
    offset = xbmin + 0.5*scalefactor
   #print(nnutils.preconditioning.__doc__)
    rc = nnutils.preconditioning(offset, scalefactor)

#-------------------------------------------------------------------------------------------
  def finalize(self):
   #print(nnutils.finalize.__doc__)
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
    return nnutils.get_analysis(self.nlon, self.nlat, self.nprs, self.nobs)

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
    rc = nnutils.reset_xb(self.xa, mlon=self.nlon, mlat=self.nlat)

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
  def reset_xb(self):
   #reset xb
    self.xa = nnutils.heavy_smooth(self.nlon, self.nlat)
    rc = nnutils.reset_xb(self.xa, mlon=self.nlon, mlat=self.nlat)
    self.prev_cost = 1.0e21

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
    obsdiff = self.obsval[:] - self.obsanl[:]

   #print('self.xa.shape = ', self.xa.shape)
   #print('self.obsanl.shape = ', self.obsanl.shape)

   #for lvl in range(50, self.npfl, 10):
    for lvl in range(50, 100, 20):
      newtitle = '%s lev %d' %(title, lvl)
      imagename = 'status_lev%d.png' %(lvl)
      pr.set_title(newtitle)
      pr.set_imagename(imagename)
      ba = self.xb[lvl,:,:]
      an = self.xa[lvl,:,:]
      dv = an - ba

      print('lvl %d: ba min: %f, max: %f' %(lvl, np.min(ba), np.max(ba)))
      print('lvl %d: an min: %f, max: %f' %(lvl, np.min(an), np.max(an)))
      print('lvl %d: dv min: %f, max: %f' %(lvl, np.min(dv), np.max(dv)))
      data = [ba, an, dv]
      pr.plot(self.lon, self.lat, data=data, obsvar=obsdiff)

#------------------------------------------------------------------------
  def plot_result(self, pr, iteration=-1):
    self.xa, self.obsanl = self.get_xaya()
    obsdiff = self.obsval[:] - self.obsanl[:]

   #for lvl in range(50, self.npfl, 10):
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

      print('lvl %d: ba min: %f, max: %f' %(lvl, np.min(ba), np.max(ba)))
      print('lvl %d: an min: %f, max: %f' %(lvl, np.min(an), np.max(an)))
      print('lvl %d: dv min: %f, max: %f' %(lvl, np.min(dv), np.max(dv)))
      data = [ba, an, dv]
      pr.plot(self.lon, self.lat, data=data, obsvar=obsdiff)
    
   #------------------------------------------------------------------------
    self.xa = nnutils.light_smooth(self.nlon, self.nlat, self.nprs)

   #for lvl in range(50, self.npfl, 10):
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

      print('lvl %d: ba min: %f, max: %f' %(lvl, np.min(ba), np.max(ba)))
      print('lvl %d: an min: %f, max: %f' %(lvl, np.min(an), np.max(an)))
      print('lvl %d: dv min: %f, max: %f' %(lvl, np.min(dv), np.max(dv)))
      data = [ba, an, dv]
      pr.plot(self.lon, self.lat, data=data, obsvar=obsdiff)
    
   #------------------------------------------------------------------------
    self.xa = nnutils.heavy_smooth(self.nlon, self.nlat, self.nprs)
    
   #for lvl in range(50, self.npfl, 10):
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
      dv = an - ba

      print('lvl %d: ba min: %f, max: %f' %(lvl, np.min(ba), np.max(ba)))
      print('lvl %d: an min: %f, max: %f' %(lvl, np.min(an), np.max(an)))
      print('lvl %d: dv min: %f, max: %f' %(lvl, np.min(dv), np.max(dv)))
      data = [ba, an, dv]
      pr.plot(self.lon, self.lat, data=data, obsvar=obsdiff)
        
#===================================================================================
