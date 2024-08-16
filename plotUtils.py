import getopt
import os, sys
import math
import numpy as np
import scipy as sp
import numpy.ma as ma

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs
from cartopy import config
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

#==================================================================================
class PlotResult(object):
  def __init__(self, debug=0, output=0):
    self.debug = debug
    self.output = output

    self.set_default()

#--------------------------------------------------------------------------------
  def plot(self, lons, lats, data=[], obsvar=[]):
    if(self.debug):
      print('obsvar min: %f, max: %f' %(np.min(obsvar), np.max(obsvar)))

    nrows = int(len(data)/2)
    ncols = 2

   #print('obsvar min: %f, max: %f' %(np.min(obsvar), np.max(obsvar)))
    scales = np.zeros((len(self.obslat),), dtype=int)
    scales[:] = 1
    colors = np.zeros((len(self.obslat),), dtype=str)
    colors[:] = 'yello'
    for n in range(len(obsvar)):
     #scales[n] = int(500.0*abs(obsvar[n]))+1
     #scales[n] = int(0.10*abs(obsvar[n]))+1
      if(obsvar[n] < -0.001):
        colors[n] = 'cyan'
      elif(obsvar[n] > 0.001):
        colors[n] = 'magenta'

   #set up the plot
    proj = ccrs.PlateCarree()

    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,
                            subplot_kw=dict(projection=proj),
                            figsize=(11,8.5))
 
   #axs is a 2 dimensional array of `GeoAxes`. Flatten it into a 1-D array
    axs=axs.flatten()

    for i in range(len(axs)):
      title = '%s' %(self.runname[i])
      axs[i].set_global()

      pvar = data[i]

     #cyclic_data, cyclic_lons = add_cyclic_point(pvar, coord=lons)

      if(i > 1):
        self.cmapname = 'bwr'
        self.clevs = np.arange(-0.5, 0.51, 0.02)
        self.cblevs = np.arange(-0.5, 0.6, 0.1)
    #else:
    #  self.cmapname = 'rainbow'
    #  self.clevs = np.arange(200.0, 312.0, 2.0)
    #  self.cblevs = np.arange(200.0, 320.0, 20.0)

     #Apply gaussian filter
      sigma_y = 3.0
      sigma_x = 2.0
      sigma = [sigma_y, sigma_x]
     #pv = sp.ndimage.filters.gaussian_filter(cyclic_data, sigma, mode='constant')
     #cs=axs[i].contourf(cyclic_lons, lats, pv, transform=proj,
      cs=axs[i].contourf(lons, lats, pvar, transform=proj,
                         levels=self.clevs, extend=self.extend,
                         alpha=self.alpha, cmap=self.cmapname)
     #marker:
      if(i > 2):
        sc = axs[i].scatter(self.obslon, self.obslat, s=scales, c=colors)

      axs[i].set_extent([-180, 180, -90, 90], crs=proj)
      axs[i].coastlines(resolution='auto', color='k')
      axs[i].gridlines(color='lightgrey', linestyle='-', draw_labels=True)

      axs[i].set_title(title)

      cb = plt.colorbar(cs, ax=axs[i], orientation=self.orientation,
                        pad=self.pad, ticks=self.cblevs)

      cb.set_label(label=self.label, size=self.size, weight=self.weight)

      cb.ax.tick_params(labelsize=self.labelsize)
      if(self.precision == 0):
        cb.ax.set_xticklabels(['{:.0f}'.format(x) for x in self.cblevs], minor=False)
      elif(self.precision == 1):
        cb.ax.set_xticklabels(['{:.1f}'.format(x) for x in self.cblevs], minor=False)
      elif(self.precision == 2):
        cb.ax.set_xticklabels(['{:.2f}'.format(x) for x in self.cblevs], minor=False)
      else:
        cb.ax.set_xticklabels(['{:.3f}'.format(x) for x in self.cblevs], minor=False)

   #Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.8,
                        wspace=0.02, hspace=0.02)

   #Add a big title at the top
    plt.suptitle(self.title)

   #fig.canvas.draw()
    plt.tight_layout()

    if(self.output):
      if(self.imagename is None):
        imagename = 't_aspect.png'
      else:
        imagename = self.imagename
      plt.savefig(imagename)
      plt.close()
    else:
      plt.show()

    self.set_default()

#--------------------------------------------------------------------------------
  def set_default(self):
    self.imagename = 'sample.png'

   #self.runname = ['Background', 'Analysis', 'BrightTemperature', 'Analysis - Background']
    self.runname = ['Background', 'Analysis', 'Analysis - Background', 'Anl-Bkg + obsdiff']

   #cmapname = coolwarm, bwr, rainbow, jet, seismic
   #self.cmapname = 'bwr'
   #self.cmapname = 'coolwarm'
    self.cmapname = 'rainbow'
   #self.cmapname = 'jet'

   #self.clevs = np.arange(-1.0, 1.02, 0.02)
   #self.cblevs = np.arange(-1.0, 1.1, 0.1)

    self.clevs = np.arange(200.0, 312.0, 2.0)
    self.cblevs = np.arange(200.0, 320.0, 20.0)

    self.extend = 'both'
    self.alpha = 0.5
    self.fraction = 0.05
    self.pad = 0.05
    self.orientation = 'horizontal'
   #self.size = 'large'
    self.size = 'medium'
    self.weight = 'bold'
    self.labelsize = 'medium'

    self.label = 'Unit (C)'
    self.title = 'Temperature Increment'

    self.precision = 1

   #self.obslon = []
   #self.obslat = []
    self.add_obs_marker = False

#--------------------------------------------------------------------------------
  def set_runname(self, name=[]):
    self.runname = name

#--------------------------------------------------------------------------------
  def set_label(self, label='Unit (C)'):
    self.label = label

#--------------------------------------------------------------------------------
  def set_title(self, title='Temperature Increment'):
    self.title = title

#--------------------------------------------------------------------------------
  def set_clevs(self, clevs=[]):
    self.clevs = clevs

#--------------------------------------------------------------------------------
  def set_cblevs(self, cblevs=[]):
    self.cblevs = cblevs

#--------------------------------------------------------------------------------
  def set_imagename(self, imagename):
    self.imagename = imagename

#--------------------------------------------------------------------------------
  def set_cmapname(self, cmapname):
    self.cmapname = cmapname

#--------------------------------------------------------------------------------
  def set_obs_lonlat(self, obslon, obslat):
    self.obslon = obslon
    self.obslat = obslat

#--------------------------------------------------------------------------------
  def switch_marker_on(self):
    self.add_obs_marker = True

#--------------------------------------------------------------------------------
  def switch_marker_off(self):
    self.add_obs_marker = False

#--------------------------------------------------------------------------------
  def set_runname(self, runname = ['Linear Observer', 'NonLinear', 'Linear Observer - NonLinear']):
    self.runname = runname

#--------------------------------------------------------------------------------
  def biasplot(self, bow, title='bias', imgname='biasplot'):
    self.imagename = '%s.png' %(imgname)

    layer,nlat,nlon = bow.shape

    y = np.linspace(0, layer-1, layer)

    n = 0
    for i in range(0, nlon, 30):
      for j in range(0, nlat, 30):
        label = 'line %d' %(n)
        x = bow[:,j,i]
        plt.plot(x, y, label=label)
        n += 1

   #Add a big title at the top
    plt.suptitle(title)

   #fig.canvas.draw()
    plt.tight_layout()

    if(self.output):
      imagename = '%s.png' %(imgname)
      plt.savefig(imagename)
      plt.close()
    else:
      plt.show()

#--------------------------------------------------------------------------------
  def weightplot(self, weight, title='weight', imgname='weightplot'):
    self.imagename = '%s.png' %(imgname)

    layer2,layer1,nlat,nlon = weight.shape

    print('layer2 %d layer1 %d nlat %d nlon %d' %(layer2,layer1,nlat,nlon))

    y = np.linspace(0, layer1-1, layer1)

    n = 0
    for i in range(0, nlon, 30):
      for j in range(0, nlat, 30):
        label = 'line %d' %(n)
        x1 = weight[:,:,j,i]
        x = np.ndarray(shape=(layer1,layer2), dtype=float)
        for l1d in range(layer1):
          x[l1d,:] = x1[:,l1d]
        plt.plot(x, y, label=label)
        n += 1

   #Add a big title at the top
    plt.suptitle(title)

   #fig.canvas.draw()
    plt.tight_layout()

    if(self.output):
      imagename = '%s.png' %(imgname)
      plt.savefig(imagename)
      plt.close()
    else:
      plt.show()

#--------------------------------------------------------------------------------
  def biasplot2(self, bow, lon, lat, obslat, obslon, obs_qc,
                title='Line', imgname='lineplot'):
    self.imagename = '%s.png' %(imgname)

    layer,nlat,nlon = bow.shape

    y = np.linspace(0, layer-1, layer)

    deltlon = lon[2] - lon[1]
    deltlat = lat[1] - lat[2]

    for n in range(20):
      i = int(obslon[n]/deltlon)
      j = int((90.0-obslat[n])/deltlat)
      if(j < 0):
        j = 0
      if(j >= nlat):
        j = nlat - 1
      label = 'line %d' %(i + j*nlon)
      x = bow[:,j,i]
      plt.plot(x, y, label=label)

   #Add a big title at the top
    plt.suptitle(title)

   #fig.canvas.draw()
    plt.tight_layout()

    if(self.output):
      imagename = '%s.png' %(imgname)
      plt.savefig(imagename)
      plt.close()
    else:
      plt.show()

#--------------------------------------------------------------------------------
  def weightplot2(self, weight, lon, lat, obslat, obslon, obs_qc,
                  title='Line', imgname='lineplot'):
    self.imagename = '%s.png' %(imgname)

    layer2,layer1,nlat,nlon = weight.shape

    print('layer2 %d layer1 %d nlat %d nlon %d' %(layer2,layer1,nlat,nlon))

    y = np.linspace(0, layer1-1, layer1)

    deltlon = lon[2] - lon[1]
    deltlat = lat[1] - lat[2]

   #for n in range(20):
   #for n in range(10):
    for n in range(1):
      i = int(obslon[n]/deltlon)
      j = int((90.0-obslat[n])/deltlat)
      if(j < 0):
        j = 0
      if(j >= nlat):
        j = nlat - 1
      label = 'line %d' %(i + j*nlon)
      x1 = weight[:,:,j,i]
      x = np.ndarray(shape=(layer1,layer2), dtype=float)
      for l1d in range(layer1):
        x[l1d,:] = x1[:,l1d]
      plt.plot(x, y, label=label)

   #Add a big title at the top
    plt.suptitle(title)

   #fig.canvas.draw()
    plt.tight_layout()

    if(self.output):
      imagename = '%s.png' %(imgname)
      plt.savefig(imagename)
      plt.close()
    else:
      plt.show()

#===================================================================================
if __name__== '__main__':
  debug = 1
  output = 0
  opts, args = getopt.getopt(sys.argv[1:], '', ['debug=', 'output='])
  for o, a in opts:
    if o in ('--debug'):
      debug = int(a)
    elif o in ('--output'):
      output = int(a)
    else:
      assert False, 'unhandled option'

#-------------------------------------------------------------------------------------------
  pr = PlotResult()

#-------------------------------------------------------------------------------------------
 #evenly sampled at 1x1 degree
  deg2arc = np.pi/180.0

  delt = 5.0
  halfdelt = delt/2

  lon = np.arange(-180.0+halfdelt, 180.0, delt)
  nlon = len(lon)
  lat = np.arange(-90.0+halfdelt, 90.0+delt, delt)
  nlat = len(lat)

  print('nlon = ', nlon)
 #print('lon = ', lon)
  print('nlat = ', nlat)
 #print('lat = ', lat)

  xa = np.ndarray((nlat, nlon), dtype=float)
  xb = 0.25*(np.random.rand(nlat, nlon) - 0.5)
  xf = np.ndarray((nlat, nlon), dtype=float)

  print('lon.shape = ', lon.shape)
  print('lat.shape = ', lat.shape)
  print('xa.shape = ', xa.shape)

  for j in range(nlat):
    for i in range(nlon):
      xa[j, i] = np.cos(2.0*lon[i]*deg2arc)*np.sin(2.0*lat[j]*deg2arc)

 #for j in range(5, nlat, 5):
 #  print('xa = ', xa[j,5:nlon:5])

  print('xa min: %f, max: %f' %(np.min(xa), np.max(xa)))

  nobs = 36*18
  obslon = 360.0*(np.random.random(nobs) - 0.5)
  obslat = 180.0*(np.random.random(nobs) - 0.5)
  obsval = 0.025*(np.random.rand(nobs) - 0.5)
 #obsval += np.sin(obslon*deg2arc)*np.cos(obslat*deg2arc)

 #print('obslon = ', obslon)
 #print('obslat = ', obslat)
 #print('obsval = ', obsval)

  print('obsval min: %f, max: %f' %(np.min(obsval), np.max(obsval)))

  title = 'Initial background and Ideal Goal Analysis'
  pr.set_title(title)

  imagename = 'initial_field.png'
  pr.set_imagename(imagename)

  pr.set_obs_lonlat(obslon, obslat)

  dv = xa - xb
  data = [xb, xa, dv]
  pr.plot(lon, lat, data=data, obsvar=obsval)

