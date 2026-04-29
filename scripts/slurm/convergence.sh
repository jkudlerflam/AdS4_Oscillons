#!/bin/bash
#SBATCH --job-name=ads4osc_conv
#SBATCH --output=convergence_%j.out
#SBATCH --error=convergence_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=typhon
#SBATCH --time=12:00:00

# Resolution convergence study
# Usage: sbatch convergence.sh
# Or: sbatch --export=ELL=4,DELTA=10,EPS=0.05 convergence.sh

ELL=${ELL:-2}
DELTA=${DELTA:-6}
EPS=${EPS:-0.05}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-96}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export LD_LIBRARY_PATH=/usr/local/gcc-7.3.0/lib64:/usr/local/intel/compilers_and_libraries_2019.2.187/linux/mkl/lib/intel64:$LD_LIBRARY_PATH

BASEDIR="$HOME/AdS4_Oscillons"
EXE="$BASEDIR/ads4osc_cli"
OUTDIR="$BASEDIR/results/convergence"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "=== Convergence study: ell=$ELL, Delta=$DELTA, eps=$EPS ==="
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo

$EXE --convergence \
    --ell $ELL --Delta $DELTA --eps $EPS \
    --verbose \
    --output "convergence_ell${ELL}_D${DELTA}_eps${EPS}.csv"

echo
echo "End: $(date)"
