#!/bin/sh
#SBATCH -t 06:00:00
#SBATCH -A gsienkf
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -p bigmem
#SBATCH -J mlcrtm
#SBATCH -e log.crtm.%J.err
#SBATCH -o log.crtm.%J.out

 source ~/pythonenv

 cd /work/noaa/gsienkf/weihuang/src/mlcrtm

 source date.env

 predatestr=$curdate
#predatestr=2022010700
 interval=6

 if [ -f saved.weight/crtm_weight_${predatestr}.nc ]
 then
   ln -sf saved.weight/crtm_weight_${predatestr}.nc pre-crtm_weight.nc
#else
#  echo "There is no good weight file: ${flnm}. exit..."
#  exit -1
 fi

 export OMP_NUM_THREADS=36
#export OMP_NUM_THREADS=1

 curdatestr=`incdate.sh ${predatestr} ${interval}`

 python mlcrtm.py --interval 6 \
	--dirname /work2/noaa/da/weihuang/EMC_cycling/jedi-cycling \
        --datestr ${curdatestr}

 if [ -f crtm_weight.nc ]
 then
   if [ ! -f saved.weight/crtm_weight_${curdatestr}.nc ]
   then
     mv crtm_weight.nc saved.weight/crtm_weight_${curdatestr}.nc
   fi
   ln -sf saved.weight/crtm_weight_${curdatestr}.nc pre-crtm_weight.nc
 fi

 if [ -f diagnosis.nc ]
 then
   if [ ! -f saved.weight/diagnosis_${curdatestr}.nc ]
   then
     mv diagnosis.nc saved.weight/diagnosis_${curdatestr}.nc
   fi
 fi

 if [ "$predatestr" -le "$enddate" ]
 then
   curdate=`incdate.sh ${predatestr} ${interval}`
   echo "export curdate=${curdate}" > date.env
   echo "export enddate=${enddate}" >> date.env
   sbatch run.slurm
 fi

