#!/usr/bin/env bash
#SBATCH -n 8
#SBATCH -p short
#SBATCH --mem 40G

set -e

# BASE_LOC = $PWD
# MYUSER = $(whoami)
# LOCALDIR = /tmp
# DATADIR = ~/data

# THISJOB = ${SLURM_JOB_NAME}
# WORKDIR = $LOCALDIR/$MYUSER/$THISJOB
# rm -rf $WORKDIR && mkdir -p $WORKDIR && cd $WORKDIR

python maxent.py 1 2 2
