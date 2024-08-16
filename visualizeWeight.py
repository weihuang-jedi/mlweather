import getopt
import os, sys
import math
import numpy as np
import scipy as sp
import numpy.ma as ma

import netCDF4

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
class VisualizeWeight(object):
  def __init__(self, debug=1, output=0,
               weightfile='crtm_weight.nc'):
   #parameters
    self.debug = debug
    self.output = output

    wf = netCDF4.Dataset(weightfile, 'r')
    self.bias1 = wf.variables['layer1bias'][:,:,:]
    self.weight1  = wf.variables['layer1wgt'][:,:,:,:]
    self.bias2 = wf.variables['layer2bias'][:,:,:]
    self.weight2  = wf.variables['layer2wgt'][:,:,:,:]
    self.lon = wf.variables['lon'][:]
    self.lat  = wf.variables['lat'][:]
    wf.close()

    self.layer2depth, self.nlev, self.nlat, self.nlon = self.weight2.shape

   #print('bias1.shape = ', self.bias1.shape)
   #print('weight1.shape = ', self.weight1.shape)
    print('bias2.shape = ', self.bias2.shape)
    print('weight2.shape = ', self.weight2.shape)
    print('self.nlon = ', self.nlon)
    print('self.nlat = ', self.nlat)
    print('self.nlev = ', self.nlev)
   #print('self.layer1depth = ', self.layer1depth)
   #print('self.layer2depth = ', self.layer2depth)

    self.set_default()

#--------------------------------------------------------------------------------
  def plot(self, lons, lats, pvar):
    print('in plot, pvar.shape = ', pvar.shape)
   #set up the plot
    proj = ccrs.PlateCarree()

    fig, axs = plt.subplots(nrows=1,ncols=1,
                            subplot_kw=dict(projection=proj),
                            figsize=(11,8.5))
 
    axs.set_global()

    cyclic_data, cyclic_lons = add_cyclic_point(pvar, coord=lons)
    title = self.runname

    cs=axs.contourf(cyclic_lons, lats, cyclic_data, transform=proj,
                    levels=self.clevs, extend=self.extend,
                    alpha=self.alpha, cmap=self.cmapname)
    axs.set_extent([-180, 180, -90, 90], crs=proj)
    axs.coastlines(resolution='auto', color='k')
    axs.gridlines(color='lightgrey', linestyle='-', draw_labels=True)

    axs.set_title(title)

    cb = plt.colorbar(cs, ax=axs, orientation=self.orientation,
                        pad=self.pad, ticks=self.cblevs)

    cb.set_label(label=self.label, size=self.size, weight=self.weight)

    cb.ax.tick_params(labelsize=self.labelsize)
    self.set_cb_ticks(cb)

   #Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.8,
                        wspace=0.02, hspace=0.02)

   #Add a big title at the top
    plt.suptitle(self.title)

   #fig.canvas.draw()
    plt.tight_layout()

    self.display(output=self.output, imagename=self.imagename)

#--------------------------------------------------------------------------------
  def display(self, output=0, imagename='sample.png'):
    if(self.output):
      plt.savefig(imagename)
      plt.close()
    else:
      plt.show()

#--------------------------------------------------------------------------------
  def set_cb_ticks(self, cb):
    if(self.precision == 0):
      cb.ax.set_xticklabels(['{:.0f}'.format(x) for x in self.cblevs], minor=False)
    elif(self.precision == 1):
      cb.ax.set_xticklabels(['{:.1f}'.format(x) for x in self.cblevs], minor=False)
    elif(self.precision == 2):
      cb.ax.set_xticklabels(['{:.2f}'.format(x) for x in self.cblevs], minor=False)
    else:
      cb.ax.set_xticklabels(['{:.3f}'.format(x) for x in self.cblevs], minor=False)

#--------------------------------------------------------------------------------
  def set_default(self):
    self.imagename = 'weight.png'

    self.runname = 'Weight'

   #cmapname = coolwarm, bwr, rainbow, jet, seismic
   #self.cmapname = 'bwr'
   #self.cmapname = 'coolwarm'
    self.cmapname = 'rainbow'
   #self.cmapname = 'jet'

   #self.clevs = np.arange(-0.01, 0.011, 0.001)
   #self.cblevs = np.arange(-0.01, 0.015, 0.005)

    self.clevs = np.arange(-0.005, 0.0051, 0.0001)
    self.cblevs = np.arange(-0.005, 0.0055, 0.0005)

    self.extend = 'both'
    self.alpha = 0.5
    self.fraction = 0.05
    self.pad = 0.05
    self.orientation = 'horizontal'
   #self.size = 'large'
    self.size = 'medium'
    self.weight = 'bold'
    self.labelsize = 'medium'

    self.label = ' '
    self.title = 'Weight'

    self.precision = 3

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
  def set_runname(self, runname = 'Weight'):
    self.runname = runname

#--------------------------------------------------------------------------------
  def biasplot(self, bow, title='Line', imgname='lineplot'):
    layer,nlat,nlon = bow.shape

    y = np.linspace(0, layer-1, layer)

    xl = np.mean(bow, axis=2)
    xm = np.mean(xl, axis=1)
    label = 'Mean'

    label = 'Mean Bias'
    x = xm[:]
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
  def biasplot1(self, bow, title='Line', imgname='lineplot'):
    layer,nlat,nlon = bow.shape

    y = np.linspace(0, layer-1, layer)

    n = 0
    i = int(nlon/2)
    j = int(nlat/2)
   #for i in range(0, nlon, 30):
   #  for j in range(0, nlat, 30):
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
  def weightplot(self, weight, title='Line', imgname='lineplot'):
    layer1,nlev,nlat,nlon = weight.shape

    print('layer1 %d nlev %d nlat %d nlon %d' %(layer1,nlev,nlat,nlon))

    y = np.linspace(0, nlev-1, nlev)

    xl = np.mean(weight, axis=3)
    xm = np.mean(xl, axis=2)
    label = 'Mean'

    self.fig, self.ax = plt.subplots(nrows=1,ncols=1,
                                     figsize=(11,8.5))
    for l1d in range(layer1):
      label = 'layer %d' %(l1d)
      x= xm[l1d,:]
      plt.plot(x, y, label=label)

   #Add a big title at the top
    plt.suptitle(title)

   #plt.legend(fontsize=16)
    plt.legend(fontsize='small')

   #fig.canvas.draw()
    plt.tight_layout()

    if(self.output):
     #imagename = '%s_layer%d.png' %(imgname, l1d)
      imagename = '%s.png' %(imgname)
      plt.savefig(imagename)
      plt.close()
    else:
      plt.show()
  
#--------------------------------------------------------------------------------
  def weightplot1(self, weight, title='Line', imgname='lineplot'):
    layer1,nlev,nlat,nlon = weight.shape

    print('layer1 %d nlev %d nlat %d nlon %d' %(layer1,nlev,nlat,nlon))

    y = np.linspace(0, nlev-1, nlev)

    i = int(nlon/2)
    j = int(nlat/2)
   #for i in range(0,nlon,30):
   #  for j in range(0,nlat,30):
    label = 'line %d' %(i + j*nlon)
    x1 = weight[:,:,j,i]
   #x = np.ndarray(shape=(layer1,layer2), dtype=float)

    for l1d in range(layer1):
      self.fig, self.ax = plt.subplots(nrows=1,ncols=1,
                                       figsize=(11,8.5))
      x = x1[l1d,:]
      plt.plot(x, y, label=label)

     #Add a big title at the top
      plt.suptitle(title)

     #fig.canvas.draw()
      plt.tight_layout()

      if(self.output):
        imagename = '%s_layer%d.png' %(imgname, l1d)
        plt.savefig(imagename)
        plt.close()
      else:
        plt.show()

#--------------------------------------------------------------------------------
  def biasplot2(self, bow, title='Line', imgname='lineplot'):
    layer,nlat,nlon = bow.shape

    y = np.linspace(0, layer-1, layer)

    x1 = np.mean(bow, axis=2)
    x = np.mean(x1, axis=1)

    plt.plot(x, y)

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
  def weightplot2(self, weight, title='Line', imgname='lineplot'):
    layer2,layer1,nlat,nlon = weight.shape

    y = np.linspace(0, layer1-1, layer1)

    xl = np.mean(weight, axis=3)
    xm = np.mean(xl, axis=2)
    label = 'Mean'
    x1 = xm[:,:]
   #x = np.ndarray(shape=(layer1,layer2), dtype=float)

    fig, ax = plt.subplots()

    for l2d in range(layer2):
      label = 'L %d' %(l2d)
      x = x1[l2d,:]
      plt.plot(x, y, label=label)

   #ax.legend(fontsize=16)
    ax.legend(fontsize='medium')

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
  def process(self):
    avg = np.mean(self.weight2, axis=1)
   #for chn in range(self.layer2depth):
    for chn in range(0):
      title = 'No Smooth Channel %d' %(chn)
      imagename = 'no-smooth_channel%d.png' %(chn)
      self.set_title(title)
      self.set_imagename(imagename)

      pvar = avg[chn,:,:]

      print('chn %d: pvar min: %f, max: %f' %(chn, np.min(pvar), np.max(pvar)))
      self.plot(self.lon, self.lat, pvar)

    zonalavg = np.mean(self.weight2, axis=3)
    chnavg = np.mean(zonalavg, axis=0)
    self.plot_meridional_section(self.lat, chnavg)

    meridianavg = np.mean(self.weight2, axis=2)
    chnavg = np.mean(meridianavg, axis=0)
    self.plot_zonal_section(self.lon, chnavg)
    
    self.biasplot2(self.bias2, title='Layer 2 Bias',
                   imgname='layer2bias')
        
    self.weightplot2(self.weight2, title='Layer 2 Weight',
                     imgname='layer2weight')
    
    self.biasplot(self.bias1, title='Layer 1 Bias',
                  imgname='layer1bias')
        
    self.weightplot(self.weight1, title='Layer 1 Weight',
                    imgname='layer1weight')

#------------------------------------------------------------------
  def plot_meridional_section(self, lat, pvar):
    nlev, nlat = pvar.shape

    self.fig, self.ax = plt.subplots(nrows=1,ncols=1,
                                     figsize=(11,8.5))

    lev = np.arange(0.0, float(nlev), 1.0)

    contfill = self.ax.contourf(lat, -lev[::-1], pvar[::-1,:], tri=True,
                                levels=self.clevs, extend=self.extend,
                                alpha=self.alpha, cmap=self.cmapname)

    cb = self.fig.colorbar(contfill, orientation=self.orientation,
                           pad=self.pad, ticks=self.cblevs)

    cb.set_label(label=self.label, size=self.size, weight=self.weight)

    cb.ax.tick_params(labelsize=self.labelsize)
    self.set_cb_ticks(cb)
    self.ax.set_title(self.title)

    major_ticks_top=np.linspace(-90,90,7)
    self.ax.set_xticks(major_ticks_top)

    intv = int(1+nlev/10)
   #major_ticks_top=np.linspace(0,nlev,intv)
   #major_ticks_top=np.linspace(0,60,7)
    major_ticks_top=np.linspace(-60,0,7)
    self.ax.set_yticks(major_ticks_top)

    minor_ticks_top=np.linspace(-90,90,19)
    self.ax.set_xticks(minor_ticks_top,minor=True)

    minor_ticks_top=np.linspace(-120,0,13)
    self.ax.set_yticks(minor_ticks_top,minor=True)

   #self.ax.grid(b=True, which='major', color='green', linestyle='-', alpha=0.5)
   #self.ax.grid(b=True, which='minor', color='green', linestyle='dotted', alpha=0.2)

    self.display(output=self.output, imagename=self.imagename)

#------------------------------------------------------------------
  def plot_meridional_section_logp(self, lat, pvar):
    nlev, nlat = pvar.shape

    self.fig, self.ax = plt.subplots(nrows=1,ncols=1,
                                     figsize=(11,8.5))

    lev = np.arange(0.0, float(nlev), 1.0)

    self.fig = self.plt.figure()
    self.ax = self.plt.subplot()

    contfill = self.ax.contourf(lat, self.logp[::-1], pvar[::-1,:], tri=True,
                                levels=self.clevs, extend=self.extend,
                                alpha=self.alpha, cmap=self.cmapname)

    cb = self.fig.colorbar(contfill, orientation=self.orientation,
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

    self.ax.set_title(self.title)

    major_ticks_top=np.linspace(-90,90,7)
    self.ax.set_xticks(major_ticks_top)

    minor_ticks_top=np.linspace(-90,90,19)
    self.ax.set_xticks(minor_ticks_top,minor=True)
    self.ax.set_xlabel('Latitude')

    self.ax.set_yticks(self.marklogp)
    self.ax.set_ylabel('Unit: hPa')

    yticklabels = []
    for p in self.markpres:
      lbl = '%d' %(int(p+0.1))
      yticklabels.append(lbl)
    self.ax.set_yticklabels(yticklabels)

    self.ax.grid(b=True, which='major', color='green', linestyle='-', alpha=0.5)
    self.ax.grid(b=True, axis='x', which='minor', color='green', linestyle='dotted', alpha=0.2)

    self.display(output=self.output, imagename=self.imagename)

#------------------------------------------------------------------
  def plot_zonal_section(self, lon, pvar):
    nlev, nlon = pvar.shape
    self.fig, self.ax = plt.subplots(nrows=1,ncols=1,
                                     figsize=(11,8.5))

    lev = np.arange(0.0, float(nlev), 1.0)

    contfill = self.ax.contourf(lon, -lev[::-1], pvar[::-1,:], tri=True,
                                levels=self.clevs, extend=self.extend,
                                alpha=self.alpha, cmap=self.cmapname)

    cb = self.fig.colorbar(contfill, orientation=self.orientation,
                           pad=self.pad, ticks=self.cblevs)

    cb.set_label(label=self.label, size=self.size, weight=self.weight)

    cb.ax.tick_params(labelsize=self.labelsize)
    self.set_cb_ticks(cb)
    self.ax.set_title(self.title)

    major_ticks_top=np.linspace(0,360,13)
    self.ax.set_xticks(major_ticks_top)

    intv = int(1+nlev/10)
   #major_ticks_top=np.linspace(0,nlev,intv)
   #major_ticks_top=np.linspace(0,60,7)
    major_ticks_top=np.linspace(-60,0,7)
    self.ax.set_yticks(major_ticks_top)

   #self.ax.grid(b=True, which='major', color='green', linestyle='-', alpha=0.5)

    minor_ticks_top=np.linspace(0,360,37)
    self.ax.set_xticks(minor_ticks_top,minor=True)

    intv = int(1+nlev/5)
    minor_ticks_top=np.linspace(-120,0,13)
    self.ax.set_yticks(minor_ticks_top,minor=True)

    self.display(output=self.output, imagename=self.imagename)

#------------------------------------------------------------------
  def plot_zonal_section_logp(self, lon, pvar):
    nlev, nlon = pvar.shape
    self.fig, self.ax = plt.subplots(nrows=1,ncols=1,
                                     figsize=(11,8.5))

    lev = np.arange(0.0, float(nlev), 1.0)

    self.fig = self.plt.figure()
    self.ax = self.plt.subplot()

    contfill = self.ax.contourf(lon, self.logp[::-1], pvar[::-1,:], tri=True,
                                levels=self.clevs, extend=self.extend,
                                alpha=self.alpha, cmap=self.cmapname)

    cb = self.fig.colorbar(contfill, orientation=self.orientation,
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

    self.ax.set_title(self.title)

    major_ticks_top=np.linspace(0,360,13,endpoint=True)
    self.ax.set_xticks(major_ticks_top)

    minor_ticks_top=np.linspace(0,360,37,endpoint=True)
    self.ax.set_xticks(minor_ticks_top,minor=True)
    self.ax.set_xlabel('Longitude')

    self.ax.set_yticks(self.marklogp)
    self.ax.set_ylabel('Unit: hPa')

    yticklabels = []
    for p in self.markpres:
      lbl = '%d' %(int(p+0.1))
      yticklabels.append(lbl)
    self.ax.set_yticklabels(yticklabels)

    self.ax.grid(b=True, which='major', color='green', linestyle='-', alpha=0.5)
    self.ax.grid(b=True, axis='x', which='minor', color='green', linestyle='dotted', alpha=0.2)

    self.display(output=self.output, imagename=self.imagename)
#===================================================================================
if __name__== '__main__':
  debug = 1
  output = 0

  dirname = '/work2/noaa/da/weihuang/EMC_cycling/jedi-cycling'
  datestr = '2022011000'

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

  weightfile = 'saved.weight/crtm_weight_%s.nc' %(datestr)

  vw = VisualizeWeight(debug=debug, output=output,
                       weightfile=weightfile)
  vw.process()

