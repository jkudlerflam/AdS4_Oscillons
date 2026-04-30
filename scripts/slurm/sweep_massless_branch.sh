#!/bin/bash
#SBATCH --job-name=ads4_branch
#SBATCH --output=branch_%A_%a.out
#SBATCH --error=branch_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=typhon
#SBATCH --time=168:00:00
#SBATCH --array=0-12
module load gcc-toolset/10 intel/2021.1

# Fixed-w stepping for massless scalar (Delta=3) sweep over ell.
# Uses --branch mode which makes steady progress in amplitude.
# Cannot navigate E_max fold, but gets deep into nonlinear regime.
#
# Array mapping:
#   0→ell=2, 1→3, 2→4, 3→5, 4→6, 5→7, 6→8, 7→9, 8→10, 9→12, 10→15, 11→20, 12→30

ELLS=(2 3 4 5 6 7 8 9 10 12 15 20 30)
DELTA=3

ELL=${ELLS[$SLURM_ARRAY_TASK_ID]}

# Angular resolution
N_THETA=$((ELL + 4))
if [ $N_THETA -lt 6 ]; then
    N_THETA=6
fi

# Radial resolution
if [ $ELL -le 10 ]; then
    N_NUC=10; N_SHELL=10
elif [ $ELL -le 20 ]; then
    N_NUC=12; N_SHELL=12
else
    N_NUC=14; N_SHELL=14
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-96}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

EXE="$HOME/AdS4_Oscillons/ads4osc_cli"
OUTDIR="$HOME/AdS4_Oscillons/results/massless"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

OUTFILE="branch_ell${ELL}_D${DELTA}_N${N_NUC}_Nth${N_THETA}.csv"
CKPTFILE="branch_ell${ELL}_D${DELTA}_N${N_NUC}_Nth${N_THETA}.ckpt"

echo "=== Branch sweep: ell=$ELL, Delta=$DELTA ==="
echo "Resolution: N_nuc=$N_NUC, N_shell=$N_SHELL, N_theta=$N_THETA"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo

# Fixed-w continuation: steady amplitude stepping
# dw_initial=0.002, adaptive up to dw_max=0.1
# w_max=1000 (effectively unlimited — convergence guard stops it)
# residual_max=1e-4 (convergence guard)
if [ -f "$CKPTFILE" ]; then
    echo "Found checkpoint $CKPTFILE — resuming"
    $EXE --branch \
        --ell $ELL --Delta $DELTA \
        --N_nuc $N_NUC --N_shell $N_SHELL --N_theta $N_THETA \
        --w_max 1000 --residual_max 1e-4 \
        --restart "$CKPTFILE" \
        --output "$OUTFILE"
else
    $EXE --branch \
        --ell $ELL --Delta $DELTA \
        --N_nuc $N_NUC --N_shell $N_SHELL --N_theta $N_THETA \
        --w_max 1000 --residual_max 1e-4 \
        --output "$OUTFILE"
fi

echo
echo "End: $(date)"
