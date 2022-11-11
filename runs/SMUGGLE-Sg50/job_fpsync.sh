#!/bin/sh
#SBATCH -p itc_cluster,shared,conroy
#SBATCH -J fpsync
#SBATCH -n 32
#SBATCH -N 1 
#SBATCH --ntasks-per-node=32
#SBATCH -o OUTPUT-fpsync.%j.out
#SBATCH -e ERROR-fpsync.%j.err
##SBATCH --exclusive
##SBATCH --contiguous
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=1000
#SBATCH -t 1-00:00           # Runtime in D-HH:MM

export lvl=$1
export INDIR="/n/holyscratch01/hernquist_lab/abeane/output-SPv12C2s-l${lvl}"
export OUTDIR="/n/holylfs05/LABS/hernquist_lab/Users/abeane/Sgrbar/runs/SMUGGLE-Purcell-vinit120-C20-M14-stars/lvl${lvl}/output"

rm $OUTDIR # delete symlink
mkdir $OUTDIR

fpsync -n $SLURM_NTASKS -o "-ax" -O "-b" ${INDIR} ${OUTDIR}/

