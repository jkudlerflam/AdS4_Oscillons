#!/bin/bash
#SBATCH --job-name=ads4osc_massless
#SBATCH --output=massless_%A_%a.out
#SBATCH --error=massless_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=typhon
#SBATCH --time=168:00:00
#SBATCH --array=0-6

# Massless scalar (Delta=3) sweep over ell
# 7 jobs: ell in {0, 2, 3, 4, 6, 8, 10}
#
# Uses pseudo-arclength continuation to navigate E_max turning point.
# Cross-check: ell=0 should match Fodor, Forgacs & Grandclement (2015).
# Prediction: E_max = 4*pi*sqrt(ell/3) for ell >> 1.
#
# Usage: sbatch sweep_massless.sh

ELLS=(0 2 3 4 6 8 10)
DELTA=3

ELL=${ELLS[$SLURM_ARRAY_TASK_ID]}

# Angular resolution:
# ell=0: spherically symmetric, N_theta=1 suffices but use 2 for safety
# ell>0: N_theta = ell + 4 (minimum 6)
if [ $ELL -eq 0 ]; then
    N_THETA=2
else
    N_THETA=$(( ELL + 4 ))
    if [ $N_THETA -lt 6 ]; then
        N_THETA=6
    fi
fi

# Radial resolution: moderate for all cases (state sizes are manageable)
if [ $ELL -le 10 ]; then
    N_NUC=10; N_SHELL=10
elif [ $ELL -le 20 ]; then
    N_NUC=12; N_SHELL=12
else
    N_NUC=14; N_SHELL=14
fi

# Arclength continuation parameters
# Start with small amplitude, let arclength stepping handle the nonlinear regime
EPS_START=0.001
EPS_SECOND=0.005
DS_INITIAL=0.01
DS_MIN=1e-7
DS_MAX=0.5
RESIDUAL_MAX=1e-4
MAX_BRANCH=2000
POINTS_AFTER_FOLD=30

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-96}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

BASEDIR="$HOME/AdS4_Oscillons"
EXE="$BASEDIR/ads4osc_cli"
OUTDIR="$BASEDIR/results/massless"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

# Compute expected state size
N_SP=$(( N_NUC + N_SHELL ))
N_T_TOTAL=3  # N_t=2 means 3 temporal points
N_STATE=$(( 8 * N_T_TOTAL * N_SP * N_THETA + 1 ))

echo "=== Massless sweep job ${SLURM_ARRAY_TASK_ID}: ell=$ELL, Delta=$DELTA ==="
echo "Resolution: N_nuc=$N_NUC, N_shell=$N_SHELL, N_theta=$N_THETA"
echo "Expected state size: ~$N_STATE"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo

OUTFILE="arclength_ell${ELL}_D${DELTA}_N${N_NUC}_Nth${N_THETA}.csv"
CKPTFILE="arclength_ell${ELL}_D${DELTA}_N${N_NUC}_Nth${N_THETA}.ckpt"

# If an arclength checkpoint exists, resume from it
if [ -f "$CKPTFILE" ]; then
    echo "Found checkpoint $CKPTFILE — resuming"
    $EXE --arclength \
        --ell $ELL --Delta $DELTA \
        --N_nuc $N_NUC --N_shell $N_SHELL --N_theta $N_THETA \
        --eps_start $EPS_START --eps_second $EPS_SECOND \
        --ds_initial $DS_INITIAL --ds_min $DS_MIN --ds_max $DS_MAX \
        --residual_max $RESIDUAL_MAX \
        --max_branch_points $MAX_BRANCH \
        --points_after_fold $POINTS_AFTER_FOLD \
        --restart "$CKPTFILE" \
        --output "$OUTFILE"
else
    $EXE --arclength \
        --ell $ELL --Delta $DELTA \
        --N_nuc $N_NUC --N_shell $N_SHELL --N_theta $N_THETA \
        --eps_start $EPS_START --eps_second $EPS_SECOND \
        --ds_initial $DS_INITIAL --ds_min $DS_MIN --ds_max $DS_MAX \
        --residual_max $RESIDUAL_MAX \
        --max_branch_points $MAX_BRANCH \
        --points_after_fold $POINTS_AFTER_FOLD \
        --output "$OUTFILE"
fi

echo
echo "End: $(date)"
