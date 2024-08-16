import getopt
import os, sys
import math
import numpy as np

from dateutil.rrule import *
from dateutil.parser import *
from datetime import *
from datetime import timedelta

#import nnutils
import utils

from neuralnetwork import NeuralNetwork

import numpy.ma as ma

#------------------------------------------------------------------------------------
def advancedate(predatestr, intv):
  year = int(predatestr[0:4])
  month = int(predatestr[4:6])
  day = int(predatestr[6:8])
  hour = int(predatestr[8:10])

  st = datetime(year, month, day, hour, 0, 0)
  dt = timedelta(hours=intv)
  ct = st + dt

 #ts = ct.strftime("%Y-%m-%dT%H:00:00Z")
  ts = ct.strftime("%Y%m%d%H")

 #print('st = ', st)
 #print('dt = ', dt)
 #print('ct = ', ct)
 #print('ts = ', ts)

  return ts

#==================================================================================
debug = 1
output = 1
errmin = 1.0e-4
iteration_max = 20
interval = 6
dirname = '/contrib/Wei.Huang/src/mlweather'
datestr = '2024080100'
akbkfile = './data/akbk127.nc4'

opts, args = getopt.getopt(sys.argv[1:], '', ['debug=', 'output=',
                          'errmin=', 'iteration_max=', 'e', 'i',
                          'dirname=', 'datestr=', 'interval='])
for o, a in opts:
  if o in ['--debug']:
    debug = int(a)
  elif o in ['--output']:
    output = int(a)
  elif o in ['--dirname']:
    dirname = a
  elif o in ['--datestr']:
    datestr = a
  elif o in ['--interval']:
    interval = int(a)
  elif o in ['--e', '--errmin']:
    errmin = float(a)
  elif o in ['--i', '--iteration_max']:
    iteration_max = int(a)
  else:
    assert False, 'unhandled option'

#-------------------------------------------------------------------------------------------
print('datestr: %s' %(datestr))
print('ymdstr: %s, hourstr: %s' %(ymdstr, hourstr))

filename = '%s/gfs_3_%s_0000_0%s.nc' %(dirname, ymdstr, hourstr)

print('filenmae: %s' %(filename))

#-------------------------------------------------------------------------------------------
#pr = PlotResult(output=output)
lat, lon, prs, tmp0 = utils.get_grid_data(filename)

nprs, nlat, nlon = tmp0.shape

for i in range(0, nlon, 20):
  print('#%d: lon %f' %(i, lon[i]))

for j in range(0, nlat, 20):
  print('#%d: lat %f' %(j, lat[j]))

for k in range(nprs):
  print('Prs %d: %f' %(k, prs[k-1]))

#-------------------------------------------------------------------------------------------
newdatestr = advancedate(datestr, interval)
ymdstr = newdatestr[0:8]
hourstr = newdatestr[8:]

print('newdatestr: %s' %(newdatestr))
print('ymdstr: %s, hourstr: %s' %(ymdstr, hourstr))

filename = '%s/gfs_3_%s_0000_0%s.nc' %(dirname, ymdstr, hourstr)

print('filenmae: %s' %(filename))

lat, lon, prs, tmp1 = utils.get_grid_data(filename)

#-------------------------------------------------------------------------------------------
NN = NeuralNetwork(lon, lat, prs, tmp0, tmp1, debug=debug)
precost = 1.0e21

NN.useSavedWeight(weightfilename='pre-mlw_weight.nc')

precost = 1.0e21
NN.set_prev_cost(precost)
#step = 0.0001
#step = 0.00375
step = 0.002
#step = 0.01
i = 0
while(i < iteration_max):
  i += 1
  curcost = NN.get_curcost()

  print('Iteration %d: precost: %e, curcost: %e, step: %e' %(i, precost, curcost, step))

  sys.stdout.flush()

  if((precost - curcost) < errmin):
    break

  precost = curcost

 #if (i % 200 == 0):
 #  NN.plot_result(pr, iteration=i)

  if(NN.train(step)):
    break

  step *= 1.10
 #step *= 0.999

NN.save_weightNbias(filename='mlw_weight.nc')

NN.saveDiagnosis(filename='diagnosis.nc')

#NN.plot_result(pr, iteration=-1)

#NN.plot_bias2(pr)
#NN.plot_weight2(pr)

NN.finalize()

