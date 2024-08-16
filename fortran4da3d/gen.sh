#!/bin/bash

 set -x

 gfortran -c nnmodules.f -ffree-form

 f2py -I/contrib/Wei.Huang/src/mlweather/fortran4da3d \
	-c nnutils.f95 -m nnutils nnmodules.o

 mv nnutils.cpython*.so ../.
