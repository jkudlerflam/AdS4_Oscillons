#!/bin/bash
#SBATCH --job-name=ads4osc_branch
#SBATCH --output=branch_%j.out
#SBATCH --error=branch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=typhon
#SBATCH --time=24:00:00

# Single branch continuation run
# Usage: sbatch branch_single.sh
# Or override: sbatch --export=ELL=4,DELTA=10,N_NUC=12 branch_single.sh

ELL=${ELL:-2}
DELTA=${DELTA:-6}
N_NUC=${N_NUC:-10}
N_SHELL=${N_SHELL:-10}
N_THETA=${N_THETA:-6}
W_MAX=${W_MAX:-0.5}
EPS_START=${EPS_START:-0.01}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-96}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export LD_LIBRARY_PATH=/usr/local/gcc-7.3.0/lib64:/usr/local/intel/compilers_and_libraries_2019.2.187/linux/mkl/lib/intel64:$LD_LIBRARY_PATH

BASEDIR="$HOME/AdS4_Oscillons"
EXE="$BASEDIR/ads4osc_cli"
OUTDIR="$BASEDIR/results/branch_ell${ELL}_D${DELTA}"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "=== Branch run: ell=$ELL, Delta=$DELTA ==="
echo "Resolution: N_nuc=$N_NUC, N_shell=$N_SHELL, N_theta=$N_THETA"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo

$EXE --branch \
    --ell $ELL --Delta $DELTA \
    --N_nuc $N_NUC --N_shell $N_SHELL --N_theta $N_THETA \
    --eps_start $EPS_START --w_max $W_MAX \
    --store --verbose \
    --output "branch_ell${ELL}_D${DELTA}_N${N_NUC}.csv"

echo
echo "End: $(date)"
