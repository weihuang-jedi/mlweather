import getopt
import os, sys
import math
import numpy as np

import nnutils
import utils

from plotUtils import PlotResult

from neuralnetwork import NeuralNetwork

import numpy.ma as ma

#==================================================================================
debug = 1
output = 0
errmin = 1.0e-4
iteration_max = 100
gridbase = './data/letkf.20201215_000000z.nc4'
obsfile = './data/sondes_obs_2020121500_m.nc4'
akbkfile = './data/akbk127.nc4'

opts, args = getopt.getopt(sys.argv[1:], '', ['debug=', 'output=',
                          'errmin', 'iteration_max', 'e', 'i'])
for o, a in opts:
  if o in ['--debug']:
    debug = int(a)
  elif o in ['--output']:
    output = int(a)
  elif o in ['--e', '--errmin']:
    errmin = float(a)
  elif o in ['--i', '--iteration_max']:
    iteration_max = int(a)
  else:
    assert False, 'unhandled option'

#-------------------------------------------------------------------------------------------
pr = PlotResult(output=output)
ak, bk = utils.get_akbk(akbkfile)
lat, lon, psf, tmp = utils.get_grid_data(gridbase)
obslon, obslat, obsprs, obsval = utils.get_obs_data(obsfile)

#-------------------------------------------------------------------------------------------
title = 'Initial background and Ideal Goal Analysis'
imagename = 'initial_field.png'

pr.set_title(title)
pr.set_imagename(imagename)
pr.set_obs_lonlat(obslon, obslat)

#------------------------------------------------------------------------
NN = NeuralNetwork(lon, lat, ak, bk, psf, tmp,
                   obslon, obslat, obsprs, obsval)
precost = 1.0e21

#NN.update()
#NN.view_status(pr, title='Status')

#step = 0.01
#NN.test_train(step)
#NN.view_status(pr, title='Test')

#step = 1.0
#NN.pre_train(step)
#NN.plot_result(pr, iteration=0)
#NN.reset_xb()
#sys.exit(-1)

step = 1.0
precost = 1.0e21
i = 0
while(i < iteration_max):
  i += 1
  curcost = NN.get_curcost()

  print('Iteration %d: precost: %e, curcost: %e' %(i, precost, curcost))

  if((precost - curcost) < errmin):
    break

  precost = curcost

  if (i % 40 == 0):
    NN.plot_result(pr, iteration=i)

  if(NN.train(step)):
    break

NN.plot_result(pr, iteration=-1)

