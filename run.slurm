#!/bin/sh
#SBATCH -t 06:00:00
#SBATCH -A ${USER}
#SBATCH -N 1
#SBATCH --ntasks-per-node=36
#SBATCH -J mlw
#SBATCH -e log.mlw.%J.err
#SBATCH -o log.mlw.%J.out

 cd /contrib/Wei.Huang/src/mlweather

 source date.env

 predatestr=$curdate
 curdatestr=${predatestr}

 python mlweather.py --interval 3 \
	--dirname /contrib/Wei.Huang/src/mlweather/gfsdata \
        --datestr ${curdatestr}

 if [ -f mlw_weight.nc ]
 then
   if [ ! -f saved.weight/mlw_weight_${curdatestr}.nc ]
   then
     mv mlw_weight.nc saved.weight/mlw_weight_${curdatestr}.nc
   fi
   ln -sf saved.weight/mlw_weight_${curdatestr}.nc pre-mlw_weight.nc
 fi

 if [ "$predatestr" -le "$enddate" ]
 then
   curdate=`incdate.sh ${predatestr} ${interval}`
   echo "export curdate=${curdate}" > date.env
   echo "export enddate=${enddate}" >> date.env
   sbatch run.slurm
 fi

