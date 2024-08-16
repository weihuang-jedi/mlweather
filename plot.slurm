#!/bin/sh
#SBATCH -t 06:00:00
#SBATCH -A epic
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -p bigmem
#SBATCH -J plot
#SBATCH -e log.plot.%J.err
#SBATCH -o log.plot.%J.out

 source ~/pythonenv

 cd /work/noaa/gsienkf/weihuang/src/mlcrtm

 dirname=/work2/noaa/da/weihuang/EMC_cycling/jedi-cycling
 datestr=2022011500

#python chknn.py --output=1 \
#       --dirname=${dirname} \
#       --datestr=${datestr} &

 python visualizeWeight.py --output=1 \
        --dirname=${dirname} \
	--datestr=${datestr}

#wait

