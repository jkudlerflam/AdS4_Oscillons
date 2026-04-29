#!/bin/bash
#SBATCH --job-name=ads4osc_sweep
#SBATCH --output=sweep_%A_%a.out
#SBATCH --error=sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=typhon
#SBATCH --time=168:00:00
#SBATCH --array=0-26

# Parameter sweep: ell x Delta
# 9 ell values x 3 Delta values = 27 jobs
#
# Usage: sbatch sweep.sh
#
# Job array mapping:
#   index = i_ell * 3 + i_Delta
#   ell   in {2, 4, 6, 8, 10, 15, 20, 30, 40}
#   Delta in {4, 6, 10}
#
# N_theta is set adaptively: N_theta = ell + 4 (minimum 6)
# This ensures sufficient angular resolution for each ell.

ELLS=(2 4 6 8 10 15 20 30 40)
DELTAS=(4 6 10)

N_DELTA=${#DELTAS[@]}
I_ELL=$(( SLURM_ARRAY_TASK_ID / N_DELTA ))
I_DELTA=$(( SLURM_ARRAY_TASK_ID % N_DELTA ))

ELL=${ELLS[$I_ELL]}
DELTA=${DELTAS[$I_DELTA]}

# Adaptive angular resolution: N_theta = ell + 4, minimum 6
N_THETA=$(( ELL + 4 ))
if [ $N_THETA -lt 6 ]; then
    N_THETA=6
fi

# Radial resolution: scale up for large ell
if [ $ELL -le 10 ]; then
    N_NUC=10; N_SHELL=10
elif [ $ELL -le 20 ]; then
    N_NUC=12; N_SHELL=12
else
    N_NUC=14; N_SHELL=14
fi

# No w_max cap — branches run to natural Newton failure (default w_max=1000 in code)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-96}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# MKL is statically linked — no LD_LIBRARY_PATH needed

BASEDIR="$HOME/AdS4_Oscillons"
EXE="$BASEDIR/ads4osc_cli"
OUTDIR="$BASEDIR/results/sweep"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

# Compute expected state size for logging
N_SP=$(( N_NUC + N_SHELL ))
N_T_TOTAL=3  # N_t=2 means 3 temporal points
N_STATE=$(( 8 * N_T_TOTAL * N_SP * N_THETA + 1 ))

echo "=== Sweep job ${SLURM_ARRAY_TASK_ID}: ell=$ELL, Delta=$DELTA ==="
echo "Resolution: N_nuc=$N_NUC, N_shell=$N_SHELL, N_theta=$N_THETA"
echo "Expected state size: ~$N_STATE"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo

OUTFILE="branch_ell${ELL}_D${DELTA}_N${N_NUC}_Nth${N_THETA}.csv"
CKPTFILE="branch_ell${ELL}_D${DELTA}_N${N_NUC}_Nth${N_THETA}.ckpt"

# If a checkpoint exists from a previous run, restart from it
if [ -f "$CKPTFILE" ]; then
    echo "Found checkpoint $CKPTFILE — resuming"
    $EXE --branch \
        --ell $ELL --Delta $DELTA \
        --N_nuc $N_NUC --N_shell $N_SHELL --N_theta $N_THETA \
        --store --verbose \
        --restart "$CKPTFILE" \
        --output "$OUTFILE"
else
    $EXE --branch \
        --ell $ELL --Delta $DELTA \
        --N_nuc $N_NUC --N_shell $N_SHELL --N_theta $N_THETA \
        --store --verbose \
        --output "$OUTFILE"
fi

echo
echo "End: $(date)"
