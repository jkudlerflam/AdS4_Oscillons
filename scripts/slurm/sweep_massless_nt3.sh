#!/bin/bash
#SBATCH --job-name=ads4_nt3
#SBATCH --output=branch_nt3_%A_%a.out
#SBATCH --error=branch_nt3_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=typhon
#SBATCH --time=168:00:00
#SBATCH --array=0-12
module load gcc-toolset/10 intel/2021.1

# Fixed-w stepping, N_t=3, massless scalar (Delta=3)
# Array: 0→ell=2, 1→3, 2→4, 3→5, 4→6, 5→7, 6→8, 7→9, 8→10, 9→12, 10→15, 11→20, 12→30

ELLS=(2 3 4 5 6 7 8 9 10 12 15 20 30)
DELTA=3
N_T=3

ELL=${ELLS[$SLURM_ARRAY_TASK_ID]}

N_THETA=$((ELL + 4))
if [ $N_THETA -lt 6 ]; then N_THETA=6; fi

if [ $ELL -le 10 ]; then
    N_NUC=8; N_SHELL=8
elif [ $ELL -le 20 ]; then
    N_NUC=10; N_SHELL=10
else
    N_NUC=12; N_SHELL=12
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-96}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

EXE="$HOME/AdS4_Oscillons/ads4osc_cli"
OUTDIR="$HOME/AdS4_Oscillons/results/massless_nt3"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

OUTFILE="branch_ell${ELL}_D${DELTA}_N${N_NUC}_Nth${N_THETA}_Nt${N_T}.csv"
CKPTFILE="branch_ell${ELL}_D${DELTA}_N${N_NUC}_Nth${N_THETA}_Nt${N_T}.ckpt"

echo "=== Nt3 sweep: ell=$ELL, Delta=$DELTA, N_t=$N_T ==="
echo "Resolution: N_nuc=$N_NUC, N_shell=$N_SHELL, N_theta=$N_THETA, N_t=$N_T"
echo "Node: $(hostname), OMP=$OMP_NUM_THREADS"
echo "Start: $(date)"

if [ -f "$CKPTFILE" ]; then
    echo "Resuming from checkpoint"
    $EXE --branch --ell $ELL --Delta $DELTA \
        --N_nuc $N_NUC --N_shell $N_SHELL --N_theta $N_THETA --N_t $N_T \
        --w_max 1000 --residual_max 1e-4 --max_iter 30 --tol 1e-7 \
        --max_branch_points 2000 \
        --restart "$CKPTFILE" --output "$OUTFILE"
else
    $EXE --branch --ell $ELL --Delta $DELTA \
        --N_nuc $N_NUC --N_shell $N_SHELL --N_theta $N_THETA --N_t $N_T \
        --w_max 1000 --residual_max 1e-4 --max_iter 30 --tol 1e-7 \
        --max_branch_points 2000 \
        --output "$OUTFILE"
fi

echo "End: $(date)"
