#!/bin/bash

 set -x

#module load gnu/9.2.0

#cd /scratch2/BMC/gsienkf/Wei.Huang/tools/mlda3d/fortran4da3d
 cd fortran4da3d

 gfortran -c nnmodules.f -ffree-form

 f2py -c nnutils.f95 -m nnutils nnmodules.o
#python -m numpy.f2py -c nnutils.f95 -m nnutils

 mv nnutils.cpython*.so ../.
