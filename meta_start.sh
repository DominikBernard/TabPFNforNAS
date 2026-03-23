#!/bin/bash
#PBS -N tabpfn_learning_curves_on_nats_bench
#PBS -q gpu
#PBS -l select=1:ncpus=2:ngpus=1:mem=48GB:scratch_local=16GB:gpu_mem=32GB
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /storage/praha1/home/bernard1/diplomka/TabPFNforNAS/tarkill_out/tabpfn_learning_curves_on_nats_bench.out

DATADIR="/storage/praha1/home/bernard1/diplomka/TabPFNforNAS/"

export TMPDIR=$SCRATCHDIR
trap 'clean_scratch' TERM EXIT

module add python/3.11

cd $SCRATCHDIR || exit 1
cp -r $DATADIR ./
cd TabPFNforNAS || exit 1

python3 -m venv venv
source venv/bin/activate

pip install --no-cache-dir --extra-index-url=https://download.pytorch.org/whl/cu118 torch
# install other dependencies in requirements.txt
pip install --no-cache-dir -r requirements.txt
pip install -e .

python3 tabpfn_predicting_learning_curves/tabpfn_on_nats_bench/pred_from_multiple_runs.py

# copy results from folder results to permanent storage
cp -r ./tabpfn_predicting_learning_curves/logs/ $DATADIR/tabpfn_predicting_learning_curves/logs/
