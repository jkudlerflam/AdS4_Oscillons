# Deploying AdS4 Oscillons to IAS Typhon

## Step 1: Upload source code

From your local machine (the folder containing this file):

```bash
rsync -avz --exclude='build*' --exclude='.git' --exclude='*.o' --exclude='*.a' \
    . typhon-login1.sns.ias.edu:~/AdS4_Oscillons/
```

Or if rsync isn't available:

```bash
tar czf ads4osc.tar.gz --exclude='build*' --exclude='.git' --exclude='*.o' --exclude='*.a' .
scp ads4osc.tar.gz typhon-login1.sns.ias.edu:~/
ssh typhon-login1.sns.ias.edu "mkdir -p ~/AdS4_Oscillons && cd ~/AdS4_Oscillons && tar xzf ~/ads4osc.tar.gz"
```

## Step 2: SSH into Typhon

```bash
ssh typhon-login1.sns.ias.edu
```

## Step 3: Load modules and build

On the login node:

```bash
cd ~/AdS4_Oscillons

# Check what's available
module avail 2>&1 | grep -i -E 'gcc|cmake|lapack|openblas|mkl'

# Load modules (adjust names to match what 'module avail' shows)
module load gcc/10.2.0    # or whatever gcc >= 7 is available
module load cmake/3.20    # or whatever cmake >= 3.14 is available
module load lapack        # or openblas or mkl

# Build
mkdir -p build_typhon && cd build_typhon
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 ads4osc_cli

# Verify it runs
./ads4osc_cli --single --eps 0.05 --N_nuc 4 --N_shell 4 --N_theta 2
```

You should see Newton converge in 3 iterations with residual ~1e-9.

**If cmake can't find LAPACK:** Try setting the path explicitly:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DLAPACK_LIBRARIES="/path/to/liblapack.so;/path/to/libblas.so"
```

Or use the shell build script which auto-detects:

```bash
bash scripts/build_and_test.sh
```

**If modules don't exist with those exact names:** Run `module avail` and look for the closest match. Common alternatives: `gcc/11.3.0`, `cmake/3.22`, `intel-mkl`, `openblas/0.3.21`.

## Step 4: Create results directory

```bash
mkdir -p ~/AdS4_Oscillons/results/{branch,convergence,sweep}
```

## Step 5: Run jobs

### Quick test (interactive, ~1 min)

```bash
cd ~/AdS4_Oscillons/build_typhon
export OMP_NUM_THREADS=4
./ads4osc_cli --single --eps 0.05 --N_nuc 6 --N_shell 6 --N_theta 4 --verbose
```

### Single branch continuation

```bash
cd ~/AdS4_Oscillons
sbatch scripts/slurm/branch_single.sh
```

Override parameters:

```bash
sbatch --export=ELL=4,DELTA=10,N_NUC=12 scripts/slurm/branch_single.sh
```

### Resolution convergence study

```bash
sbatch scripts/slurm/convergence.sh
# Or with parameters:
sbatch --export=ELL=2,DELTA=6,EPS=0.05 scripts/slurm/convergence.sh
```

### Full parameter sweep (15 jobs: ell=2,4,6,8,10 x Delta=4,6,10)

```bash
sbatch scripts/slurm/sweep.sh
```

This submits a SLURM job array with 15 tasks. Each gets 24 cores and 64 GB RAM.

## Step 6: Monitor jobs

```bash
squeue -u $USER              # see your running/pending jobs
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS  # job details
tail -f results/sweep/sweep_JOBID_0.out   # follow output
scancel JOBID                # cancel a job
scancel JOBID_[5-14]         # cancel specific array tasks
```

## Step 7: Retrieve results

From your local machine:

```bash
rsync -avz typhon-login1.sns.ias.edu:~/AdS4_Oscillons/results/ ./results/
```

## Resource estimates

| Run type | Resolution | State size | Time per Newton iter | Est. wall time |
|----------|-----------|-----------|---------------------|---------------|
| Single solve | N=4, Nθ=2 | 649 | ~0.4s (1 core) | 2s |
| Single solve | N=8, Nθ=4 | ~2500 | ~15s (1 core) | 2 min |
| Single solve | N=10, Nθ=6 | ~5000 | ~2 min (1 core), ~6s (24 cores) | 1 min |
| Branch (30 pts) | N=10, Nθ=6 | ~5000 | ~6s (24 cores) | ~1 hr |
| Branch (30 pts) | N=12, Nθ=8 | ~10000 | ~30s (24 cores) | ~6 hr |
| Full sweep (15 jobs) | mixed | mixed | — | ~6-24 hr |

The Jacobian build dominates: it's n residual evaluations per Newton iteration, parallelized over OMP threads. With 24 cores on Cascade Lake, expect ~24x speedup on the Jacobian.

## Troubleshooting

**"module: command not found"**: Add `source /etc/profile.d/modules.sh` to your `.bashrc`.

**LAPACK not found**: Check `ldconfig -p | grep lapack`. If it's Intel MKL, you may need `module load intel-mkl` and set `MKLROOT`.

**Out of memory**: The dense Jacobian is n×n doubles. At n=10000, that's 800 MB. The 384 GB/node limit is generous, but if you go to n=20000+ you'll need the 64 GB allocation in the SLURM script bumped up.

**Slow convergence at high resolution**: If Newton needs >10 iterations, the initial guess quality may be poor. Try reducing `--eps_start` or `--w_max` to take smaller continuation steps.
