# Aer - high performance quantum circuit simulation for Qiskit

[![License](https://img.shields.io/github/license/Qiskit/qiskit-aer.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)
[![Build](https://github.com/Qiskit/qiskit-aer/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/Qiskit/qiskit-aer/actions/workflows/build.yml)
[![Tests](https://github.com/Qiskit/qiskit-aer/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Qiskit/qiskit-aer/actions/workflows/tests.yml)
[![](https://img.shields.io/github/release/Qiskit/qiskit-aer.svg?style=popout-square)](https://github.com/Qiskit/qiskit-aer/releases)
[![](https://img.shields.io/pypi/dm/qiskit-aer.svg?style=popout-square)](https://pypi.org/project/qiskit-aer/)

**Aer** is a high performance simulator for quantum circuits written in Qiskit, that includes realistic noise models.

> **This is a fork.** The `tensor-networks-slicer` branch of
> [QuantuMikey/qiskit-aer](https://github.com/QuantuMikey/qiskit-aer/tree/tensor-networks-slicer)
> adds tensor network simulation on **AMD GPUs** (ROCm). It is experimental and
> is not part of the official [Qiskit/qiskit-aer](https://github.com/Qiskit/qiskit-aer)
> project. The badges above refer to upstream. The upstream README follows
> after the fork notes below.

## About this fork

### Why it exists

In upstream Aer, `method="tensor_network"` does not do the tensor network work
itself: it offloads path search, slicing and contraction to NVIDIA cuQuantum
(cuTensorNet), which is closed-source and only runs on NVIDIA GPUs. ROCm builds
of upstream Aer support statevector simulation but have no tensor network
backend.

This fork implements the tensor network method natively for AMD GPUs. It was
built to run large tensor network simulations on
[LUMI](https://www.lumi-supercomputer.eu/), whose GPU partition uses AMD
Instinct MI250X GPUs, and is intended for other AMD-based systems as well. The
fork itself is controlled entirely by environment variables and has no
dependence on a particular scheduler or site; the LUMI-specific scripts are
collected in the [appendix](#appendix-running-on-lumi).

### What is different from upstream

| Area | Upstream qiskit-aer | This fork |
|---|---|---|
| Tensor network on AMD GPUs | Not available | Native ROCm contractor: each pairwise contraction runs as one hipBLAS ZGEMM (CGEMM in single precision), with the fork's own kernels handling operand layout |
| Contraction path search | Done inside cuTensorNet | [cotengra](https://github.com/jcmgray/cotengra) hyper-optimizer, with optional Mt-KaHyPar and cotengrust |
| Large networks | Slicing handled inside cuTensorNet | Memory-aware slicing, with slices distributed across MPI ranks |
| Path search under MPI | Handled inside cuTensorNet | Search trials sharded across ranks; best plan (lowest FLOP count after slicing) chosen collectively |
| Plan reuse | None | Contraction plans saved to disk and replayed in later runs, at any number of GPUs |
| CPU-only path search | n/a | Search on CPU nodes, contraction later on GPU nodes |
| Slice-invariant hoisting | n/a | Steps that do not depend on the sliced indices are computed once and skipped on every later slice |
| GPU capability query | n/a | `AerSimulator.gpu_device_count()` and `AerSimulator.tensor_network_gpu_available()` |
| Statevector on AMD GPUs | 32-bit grid dimension overflow at large qubit counts | Fixed in `ExecuteSum`/`ExecuteSum2` and blocked-gate kernels |

Note: the build flag and macro are still named `AER_ENABLE_HIPTENSOR` /
`AER_HIPTENSOR` for historical reasons. The contraction backend is hipBLAS;
hipTensor is no longer linked.

## Requirements

| Component | Version | Notes |
|---|---|---|
| ROCm with hipBLAS | 6.4.2 / hipBLAS 2.4.0 | the only GEMM backend measured |
| Python | 3.12 | tested with |
| **cotengra** | **0.7.5, exact** | **Hard pin.** The fork patches cotengra's internals (the search memory cap rebinds module-level objects and import-time preset instances; the forge calls `ContractionTree.from_path(..., ssa_path=...)` as 0.7.5 defines it). Another release needs changes to the fork's source. |
| cotengrust | 0.2.1 | Rust greedy path search; used outside the memory cap |
| optuna | 3.6.2 | default `optlib` for the hyper search |
| cmaes | 0.13.1 | alternative `optlib` |
| kahypar | 1.3.7 | kahypar-class trial methods in cotengra's default pool |
| mtkahypar | 1.6.2 | the `mt-kahypar` search method used for the largest networks |
| mpi4py | 4.0.3 | multi-GPU runs |
| qiskit | 2.1.2 | tested with |
| numpy | 2.2.6 | tested with |

For multi-GPU runs, Aer must be linked against a GPU-aware MPI. On a Cray
system this means Cray MPICH with its GTL library (`libmpi_gtl_hsa.so`); on
other systems a GPU-aware OpenMPI or MPICH built with ROCm support. Without
GPU-aware MPI the fork still runs on a single GPU.

A reference container build (Singularity, Cray MPICH, all packages above) is
in [QuantuMikey/CSC_QT_simulations_container_builder](https://github.com/QuantuMikey/CSC_QT_simulations_container_builder),
subdirectory `ccpe-extensions-cray-qiskit-aer-patch-tensor-networks`. It is
LUMI-specific, but its stage definitions list every build step.

## Building

Enable the tensor network backend in a ROCm build with:

```bash
export AER_ENABLE_HIPTENSOR=1
```

Then follow the ROCm build instructions in the
[contributing guide](CONTRIBUTING.md). The build stops with an error if
cotengra or hipBLAS is not found. Usage is the same as upstream:

```python
from qiskit_aer import AerSimulator

sim = AerSimulator(method="tensor_network", device="GPU")
```

Small circuits need nothing else. The rest of this section is about networks
too large for one GPU.

## How large networks are run

The pipeline has two halves that run separately:

- **The forge** (path search). cotengra searches for a cheap contraction
  order, the slicer cuts indices until every intermediate fits the per-slice
  memory target, and the result is saved as a **plan file**: the step list,
  the sliced indices and the exact FLOP count. The forge needs no GPU and runs
  on CPU nodes.
- **The replay** (contraction). Loads a plan file and executes it on the
  GPUs. Slices are the unit of distribution: each MPI rank takes a contiguous
  range and the partial results are reduced at the end. The same plan file
  replays at any number of ranks.

Two facts about memory and threads decide how to size the runs:

- **Device memory.** Before contracting, the fork lays out a pool holding every
  intermediate of a slice by liveness. That pool is about **3× the per-slice
  peak**, so the slice target should be about **1/8 of device memory** (8 GiB
  on a 64 GB GPU). Slice-invariant hoisting pins a further "frontier" of
  reused results into the pool for the whole run; its size depends on the
  plan, not on the slice target, and on the largest networks it no longer
  fits, in which case hoisting is turned off (`AER_TN_HOIST=0`). The
  `[AER_TN_INVARIANT]` line printed at setup gives both numbers.
- **Search threads.** The forge has two stages: Mt-KaHyPar partitioning
  trials, which are multithreaded, and cotengra's slicing/reconfiguration,
  which is single-threaded Python and takes about half the search time on
  large networks. More threads plateau early: on 128-core EPYC nodes, 128
  threads ran the partitioning 1.6× slower than 16. Run several searches with
  different seeds in parallel instead; plan cost varies by orders of magnitude
  between seeds, and the best plan is what gets replayed.

### Step 1: forge on a CPU node

One process, 16 physical cores, no GPU. Repeat with different seeds; each run
saves a plan file named after its seed, slice target and FLOP count, and the
`captured plan to` line in its log gives the full path.

```bash
export AER_TN_PLAN_FORGE=1              # serial search that banks a plan
export AER_TN_FORGE_ONLY=1              # exit before any GPU work
export AER_TN_PLAN_DIR=$PWD/plans       # where plans are saved
export AER_TN_FORGE_BUDGET_MB=65226     # free memory of one replay GPU, MB
export AER_TN_SLICE_TARGET_BYTES=8589934592   # 8 GiB: ~1/8 of a 64 GB GPU
export AER_TN_MIN_SLICES_PER_RANK=64    # slice floor, so wide replays have work
export AER_TN_SLICE_RECONF_EVERY=1      # reconfigure the tree after every cut
export AER_TN_PATH_PRESET=hyper
export AER_TN_PATH_METHODS=mt-kahypar
export AER_TN_MTKAHYPAR_THREADS=16
export AER_TN_PATH_SEED=12142           # change per run
export AER_TN_PATH_MAX_REPEATS=16       # trials per search
export AER_TN_PATH_MAX_TIME=14400       # search wall cap, s
export AER_TN_LOG_TIMES=1               # phase timings in the log
export AER_TN_PATH_VERBOSE=1            # search and plan-cost lines
export MALLOC_ARENA_MAX=2
python3 -u tree_row.py
```

A forge run ends with the backend raising a tagged error instead of
contracting; that is the designed exit on a node with no GPU. The log's
`plan cost:` line reports the plan's total FLOPs and slice count.

### Step 2: probe on one GPU

Before committing a large allocation, replay the plan on **one GPU with a
short wall limit** (5 minutes). This answers the two questions that decide the
real run:

- **Does it fit?** The pool is allocated at setup. An out-of-memory here means
  the plan needs `AER_TN_HOIST=0`, or a smaller slice target at the forge.
- **How long is one slice?** After the first slice, every rank extrapolates
  `first-slice time × remaining slices` against the job's remaining wall time
  (`SLURM_JOB_END_TIME` under SLURM, or `AER_TN_WALL_BUDGET_S`) and aborts
  with that estimate if it does not fit. On a short wall this refusal is the
  intended outcome: the message carries the measured per-slice time, which
  sizes the real run. `AER_TN_WALL_GATE=warn` reports without aborting.

```bash
export AER_TN_PLAN_FILE=/abs/path/plans/plan_<key>.s12142.t8g.<...>.ctg
export AER_TN_GPU_PERSIST=1
export AER_TN_PROFILE=1                 # per-rank contraction timing
export AER_TN_MEMORY_VERBOSE=1          # pool size before allocation
export MALLOC_ARENA_MAX=2
python3 -u tree_row.py
```

Do not enable GPU-aware MPI flags for a single-process run; some MPI stacks
abort at init when they are set with one task.

### Step 3: replay at width

Launch one process per GPU with GPU-aware MPI, replaying the same plan file.
Choose the rank count so that slices divide evenly and
`slices per rank × per-slice time` fits the wall. Keep `AER_TN_VERBOSE` off:
at 2048 ranks its per-step lines pushed ~25 million lines through the
launcher's stdio and stretched setup from 30 seconds to over 7 minutes.

The per-rank contraction time is printed on the `t_contract_ms` line
(`AER_TN_PROFILE=1`); read the mean across ranks, since the spread is a few
percent.

### Settings reference

| Variable | Purpose |
|---|---|
| `AER_TN_PLAN_FORGE=1` | Serial search per rank that banks a plan into the plan directory |
| `AER_TN_FORGE_ONLY=1` | Exit after banking the plan, before any GPU work; allows the forge on nodes with no GPU |
| `AER_TN_PLAN_DIR` | Directory of saved plans, one per network topology; keeps the best plan per topology |
| `AER_TN_PLAN_FILE` | A single plan file to replay; no search runs |
| `AER_TN_FORGE_BUDGET_MB` | Free device memory of the GPU the plan will replay on, in MB; required by a forge on a node with no GPU |
| `AER_TN_SLICE_TARGET_BYTES` | Per-slice memory target; lower values slice more aggressively. About 1/8 of device memory, because the pool is ~3× the per-slice peak |
| `AER_TN_MIN_SLICES_PER_RANK` | Minimum slices per rank, applied at the forge; set it to the widest replay you will want |
| `AER_TN_SLICE_RECONF_EVERY` | Reconfigure the contraction tree after every N cuts during slicing (1 recommended for deep networks) |
| `AER_TN_PATH_MAX_REPEATS` | Number of search trials; under MPI the total across ranks |
| `AER_TN_PATH_MAX_TIME` | Search wall cap in seconds |
| `AER_TN_PATH_PRESET` / `AER_TN_PATH_METHODS` | cotengra preset and method list (e.g. `hyper`, `mt-kahypar`) |
| `AER_TN_MTKAHYPAR_THREADS` | Threads for Mt-KaHyPar trials |
| `AER_TN_PATH_SEED` | Search seed; different seeds give different plans |
| `AER_TN_HOIST` | Slice-invariant hoisting; on by default, `0` to disable when the pinned frontier does not fit |
| `AER_TN_WALL_GATE` | First-slice wall gate: unset/`1` aborts, `warn` reports only, `0` disables |
| `AER_TN_GPU_PERSIST=1` | Keep device context and pool alive across contractions in one process |
| `AER_TN_PROFILE`, `AER_TN_LOG_TIMES`, `AER_TN_PATH_VERBOSE`, `AER_TN_MEMORY_VERBOSE` | Diagnostics: per-phase timings, search lines, pool size |
| `AER_TN_VERBOSE` | Per-step setup logging; keep off for large runs |

Further variables (`AER_TN_*`) are documented in comments in
`src/simulators/tensor_network/path_optimizer.hpp`.

## Validation: Table I of arXiv:2107.00677, p = 1 to 9

The fork was validated by reproducing Table I of
[arXiv:2107.00677](https://arxiv.org/abs/2107.00677) (Wurtz and Lykov,
fixed-angle QAOA for MaxCut on 3-regular graphs). Row p of the table is the
guaranteed cut fraction on the depth-p tree subgraph, C_p = (1 − ⟨Z₀Z₁⟩) / 2 on
the root edge, with N = 2^(p+2) − 2 qubits. At p = 9 the circuit has 2046
qubits. Every row from p = 1 to p = 9 was computed with
`method="tensor_network"` on MI250X GPUs and compared with the published
constant.

### The script

The same script runs the forge and the replay; the environment decides which.
The fixed angles come from
[danlkv/fixed-angle-QAOA](https://github.com/danlkv/fixed-angle-QAOA)
(3-regular graphs, key `"3"`), stored next to the script as
`fixed_angles_3reg.json` in the form
`{"9": {"gamma": [...], "beta": [...], "AR": ...}, ...}`.

```python
# tree_row.py
import json, math, os, sys

# Import order matters: qiskit_aer initializes MPI at import time, so it
# comes first and mpi4py attaches afterwards without initializing.
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI


def tree_graph(p):
    """Depth-p tree: two roots joined by edge (0, 1), every internal vertex
    has two children. N = 2^(p+2) - 2 vertices."""
    edges, frontier, nxt = [(0, 1)], [0, 1], 2
    for _ in range(p):
        new = []
        for v in frontier:
            for _ in range(2):
                edges.append((v, nxt))
                new.append(nxt)
                nxt += 1
        frontier = new
    return nxt, edges


def qaoa_circuit(n, edges, gammas, betas):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for u, v in edges:
            qc.rzz(-g, u, v)
        qc.rx(2 * b, range(n))
    return qc


rank = MPI.COMM_WORLD.Get_rank()
p = int(os.environ["TREE_P"])
here = os.path.dirname(os.path.abspath(__file__))
row = json.load(open(os.path.join(here, "fixed_angles_3reg.json")))[str(p)]
n, edges = tree_graph(p)
qc = qaoa_circuit(n, edges, row["gamma"], row["beta"])
qc.save_expectation_value(Pauli("ZZ"), [0, 1], label="zz")

sim = AerSimulator(method="tensor_network", device="GPU")
sim.set_max_qubits(n)  # the default cap is the statevector limit, meaningless for TN
result = sim.run(transpile(qc, sim, optimization_level=0), shots=1).result()

# A forge-only run saves its plan and then returns an unsuccessful result
# by design; it does not contract.
if os.environ.get("AER_TN_FORGE_ONLY") == "1":
    ok = "plan capture complete at setup" in str(result.status)
    if rank == 0:
        print("forge OK" if ok else f"forge FAILED: {result.status}")
    sys.exit(0 if ok else 1)

if rank == 0:
    zz = float(result.data(0)["zz"])
    c_p = (1 - zz) / 2
    print(f"p={p} N={n} ZZ={zz:+.15f} C_p={c_p:.15f} "
          f"table={row['AR']:.15f} diff={abs(c_p - row['AR']):.3e}")
```

### What was run

- **p = 1 to 7**: one forge per depth, replayed on 4 GPUs.
- **p = 8**: forge, then a seed sweep; the best plan (4.0e14 FLOPs, 64
  slices) replayed on 8 GPUs with hoisting on.
- **p = 9**: 32 forges at the 8 GiB slice target (24 distinct seeds, 8 of
  them repeated with 128 threads as a thread-count comparison) plus 8 at a 12
  GiB target. Each search took 7–11 minutes. Plan cost varied by a factor of
  10⁴ across seeds; the best (seed 12142, 1.1e18 FLOPs, 16,384 slices) was
  probed on one GPU, which showed the pool fits only with `AER_TN_HOIST=0`
  (28 GiB pool; the 27 GB hoist frontier would exceed the 64 GB GPU), and
  measured 36–39 s per slice. It was then replayed on 2048 GPUs at 8 slices
  per rank.

### Results

| p | Qubits | GPUs (MI250X GCDs) | Slices | Run time | GPU-hours | Diff vs. paper |
|---|---|---|---|---|---|---|
| 1 | 6 | 4 | 16 | 12 s | 0.01 | 4e-8 |
| 2 | 14 | 4 | 16 | 11 s | 0.01 | 1e-7 |
| 3 | 30 | 4 | 16 | 11 s | 0.01 | 3e-7 |
| 4 | 62 | 4 | 16 | 12 s | 0.01 | 7e-7 |
| 5 | 126 | 4 | 16 | 12 s | 0.01 | 1e-6 |
| 6 | 254 | 4 | 128 | 16 s | 0.02 | 3e-6 |
| 7 | 510 | 4 | 16 | 20 s | 0.02 | 1e-5 |
| 8 | 1022 | 8 | 64 | 34 s | 0.08 | 2e-5 |
| 9 | 2046 | 2048 | 16,384 | 325 s | 185 | 5e-5 |

- Each MI250X has two GCDs; each GCD appears to the software as one GPU.
- "Run time" is the simulator's own timer (transpile, contraction and
  reductions); container start is excluded. GPU-hours are run time × GPUs.
  The p = 9 job was billed as 6 min 13 s of wall time on 256 nodes, about 212
  GPU-hours. The forges are not included; they ran on CPU nodes.
- "Diff vs. paper" is the absolute difference between the computed C_p and the
  published value. The published values carry their own optimizer residual,
  so differences below 1e-4 are a match.
- For p ≤ 5 and p = 7 the 16 slices are the forge's distribution floor, not a
  memory requirement; from p = 6 the slice count is set by the memory
  envelope. For p ≤ 5 the run time is almost entirely fixed per-job overhead.
- p = 8 ran with hoisting on; p = 9 with hoisting off.

### Status

Experimental. Validated up to p = 9 (2046 qubits) as above. Larger p was not
attempted: cost grew well over 1000× from p = 8 to p = 9, and the p = 9 run
was already at the memory limit of a 64 GB GPU.

Related: [Quantum simulations on LUMI with Qiskit Aer](https://fiqci.fi/publications/2025-04-01-LUMI-quantum-simulations-qiskit-aer) (FiQCI blog).

## Appendix: running on LUMI

The exact commands used for the p = 9 row. LUMI uses SLURM, Singularity, Cray
MPICH and an `/appl`-provided GPU-affinity wrapper; the `SINGULARITYENV_*`
prefix passes a variable into the container without the prefix.

**Partitions.** The forge ran on `small` (LUMI-C, CPU-only, billed per core).
The single-GPU probe ran on `small-g` (partial node). The full replay ran on
`standard-g` (whole nodes, 8 GCDs each, billed as 8 GPU-hours per node-hour).

**`standard-g` shape.** Use `--ntasks` (not `--ntasks-per-node`) and
`--gpus-per-task=1` (not `--gpus-per-node`); otherwise the GPU-affinity
wrapper can hide the GPUs from the ranks. Bind one rank per GCD with the
LUMI CPU mask below. On `small-g`, strip `SLURM_NTASKS_PER_NODE` inside the
`srun` so the wrapper's device override cannot fire on a partial node.

**GPU-aware MPI.** Set the Cray MPICH GPU flags only when `ntasks > 1`. With
one task the wrapper aborts at MPI init with "GPU_SUPPORT_ENABLED is
requested, but GTL library is not linked".

**Container image.** Stripe the image before a many-node run; a 16 GB image at
stripe count 1 is read by every node from one OST. The LUMI-recommended
layout is `lfs migrate -E 256M -C 1 -S 4M -E 16G -C 4 -S 4M -E EOF -C 8 -S 4M <image>.sif`.

### `forge.sh`

```bash
#!/bin/bash
#SBATCH --job-name=forge
#SBATCH --account=<project>
#SBATCH --nodes=1
#SBATCH --ntasks=1

P="$1"
SIF=<container>.sif
PLANS=$PWD/plans
mkdir -p "$PLANS"

export SINGULARITYENV_TREE_P="$P"
export SINGULARITYENV_AER_TN_PLAN_FORGE=1
export SINGULARITYENV_AER_TN_FORGE_ONLY=1
export SINGULARITYENV_AER_TN_PLAN_DIR="$PLANS"
export SINGULARITYENV_AER_TN_MIN_SLICES_PER_RANK=64
export SINGULARITYENV_AER_TN_FORGE_BUDGET_MB=65226   # free memory of one MI250X GCD
export SINGULARITYENV_AER_TN_SLICE_RECONF_EVERY=1
export SINGULARITYENV_AER_TN_LOG_TIMES=1
export SINGULARITYENV_AER_TN_PATH_VERBOSE=1
export SINGULARITYENV_MALLOC_ARENA_MAX=2

srun --ntasks=1 singularity exec "$SIF" python3 -u tree_row.py
```

Command used for p = 9 (16 physical cores; `--hint=nomultithread` keeps SMT
siblings off the allocation). Submitted once per seed:

```bash
sbatch --partition=small --cpus-per-task=16 --hint=nomultithread \
  --mem=64G --time=02:00:00 \
  --export=ALL,SINGULARITYENV_AER_TN_PATH_PRESET=hyper,\
SINGULARITYENV_AER_TN_PATH_METHODS=mt-kahypar,\
SINGULARITYENV_AER_TN_MTKAHYPAR_THREADS=16,\
SINGULARITYENV_AER_TN_PATH_SEED=12142,\
SINGULARITYENV_AER_TN_SLICE_TARGET_BYTES=8589934592,\
SINGULARITYENV_AER_TN_PATH_MAX_TIME=14400,\
SINGULARITYENV_AER_TN_PATH_MAX_REPEATS=16 \
  forge.sh 9
```

### `replay.sh`

```bash
#!/bin/bash
#SBATCH --job-name=replay
#SBATCH --account=<project>
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7

P="$1"
PLAN="$2"
SIF=<container>.sif
WRAP=/appl/local/quantum/qiskit/run-singularity-with-gpu-affinity

if [ "${SLURM_NTASKS:-1}" -gt 1 ]; then
  export MPICH_GPU_SUPPORT_ENABLED=1
  export MPICH_GPU_IPC_CACHE_MAX_SIZE=100
  export MPICH_GPU_IPC_THRESHOLD=524288
  export MPICH_OFI_NIC_POLICY=GPU
else
  export MPICH_GPU_SUPPORT_ENABLED=0
fi

MASK=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000
if [ "$SLURM_JOB_PARTITION" = "standard-g" ]; then
  SRUN=(srun --cpu-bind=$MASK)
else
  SRUN=(srun --ntasks="$SLURM_NTASKS" env -u SLURM_NTASKS_PER_NODE)
fi

export SINGULARITYENV_TREE_P="$P"
export SINGULARITYENV_AER_TN_PLAN_FILE="$PLAN"
export SINGULARITYENV_AER_TN_GPU_PERSIST=1
export SINGULARITYENV_AER_TN_PROFILE=1
export SINGULARITYENV_AER_TN_RG_THREADS=7
export SINGULARITYENV_MALLOC_ARENA_MAX=2

"${SRUN[@]}" "$WRAP" "$SIF" python3 -u tree_row.py
```

Single-GPU probe (5 minutes; ends with the wall gate's measured per-slice
time, or an out-of-memory if the plan does not fit):

```bash
sbatch --partition=small-g --ntasks=1 --gpus-per-task=1 --cpus-per-task=7 \
  --mem=60G --time=00:05:00 \
  --export=ALL,SINGULARITYENV_AER_TN_HOIST=0,SINGULARITYENV_AER_TN_MEMORY_VERBOSE=1 \
  replay.sh 9 /abs/path/plans/plan_<key>.s12142.t8g.<...>.ctg
```

Full replay (256 nodes, 2048 GCDs, 8 slices per rank, about 6 minutes wall):

```bash
sbatch --partition=standard-g --nodes=256 --ntasks=2048 \
  --gpus-per-task=1 --cpus-per-task=7 --mem=480G --time=00:15:00 \
  --export=ALL,SINGULARITYENV_AER_TN_HOIST=0 \
  replay.sh 9 /abs/path/plans/plan_<key>.s12142.t8g.<...>.ctg
```

Result line from that job:

```
[ROW] p=9 N=2046 ranks=2048 ZZ=-0.746948424903324 C_p=0.873474212451662 C_p_table=0.873521564002300 diff=4.735e-05 t_run_s=324.763 verdict=PASS
```

---

# Upstream Qiskit Aer README

The sections below are the unchanged upstream README. Note that
`pip install qiskit-aer` installs upstream Aer, which does not include this
fork's AMD tensor network backend; build from source as described in
[Building](#building) instead.

## Installation

We encourage installing Aer via the pip tool (a python package manager):

```bash
pip install qiskit-aer
```

Pip will handle all dependencies automatically for us, and you will always install the latest (and well-tested) version.

To install from source, follow the instructions in the [contribution guidelines](CONTRIBUTING.md).

## Installing GPU support

In order to install and run the GPU supported simulators on Linux, you need CUDA&reg; 11.2 or newer previously installed.
CUDA&reg; itself would require a set of specific GPU drivers. Please follow CUDA&reg; installation procedure in the NVIDIA&reg; [web](https://www.nvidia.com/drivers).

If you want to install our GPU supported simulators, you have to install this other package:

```bash
pip install qiskit-aer-gpu
```

The package above is for CUDA&reg; 12, so if your system has CUDA&reg; 11 installed, install separate package:
```bash
pip install qiskit-aer-gpu-cu11
```

This will overwrite your current `qiskit-aer` package installation giving you
the same functionality found in the canonical `qiskit-aer` package, plus the
ability to run the GPU supported simulators: statevector, density matrix, and unitary.

**Note**: This package is only available on x86_64 Linux. For other platforms
that have CUDA support, you will have to build from source. You can refer to
the [contributing guide](CONTRIBUTING.md#building-with-gpu-support)
for instructions on doing this.

## Simulating your first Qiskit circuit with Aer
Now that you have Aer installed, you can start simulating quantum circuits using primitives and noise models. Here is a basic example:

```
$ python
```

```python
from qiskit import transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

sim = AerSimulator()
# --------------------------
# Simulating using estimator
#---------------------------
from qiskit_aer.primitives import EstimatorV2

psi1 = transpile(RealAmplitudes(num_qubits=2, reps=2), sim, optimization_level=0)
psi2 = transpile(RealAmplitudes(num_qubits=2, reps=3), sim, optimization_level=0)

H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
H2 = SparsePauliOp.from_list([("IZ", 1)])
H3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

theta1 = [0, 1, 1, 2, 3, 5]
theta2 = [0, 1, 1, 2, 3, 5, 8, 13]
theta3 = [1, 2, 3, 4, 5, 6]

estimator = EstimatorV2()

# calculate [ [<psi1(theta1)|H1|psi1(theta1)>,
#              <psi1(theta3)|H3|psi1(theta3)>],
#             [<psi2(theta2)|H2|psi2(theta2)>] ]
job = estimator.run(
    [
        (psi1, [H1, H3], [theta1, theta3]),
        (psi2, H2, theta2)
    ],
    precision=0.01
)
result = job.result()
print(f"expectation values : psi1 = {result[0].data.evs}, psi2 = {result[1].data.evs}")

# --------------------------
# Simulating using sampler
# --------------------------
from qiskit_aer.primitives import SamplerV2
from qiskit import QuantumCircuit

# create a Bell circuit
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
bell.measure_all()

# create two parameterized circuits
pqc = RealAmplitudes(num_qubits=2, reps=2)
pqc.measure_all()
pqc = transpile(pqc, sim, optimization_level=0)
pqc2 = RealAmplitudes(num_qubits=2, reps=3)
pqc2.measure_all()
pqc2 = transpile(pqc2, sim, optimization_level=0)

theta1 = [0, 1, 1, 2, 3, 5]
theta2 = [0, 1, 2, 3, 4, 5, 6, 7]

# initialization of the sampler
sampler = SamplerV2()

# collect 128 shots from the Bell circuit
job = sampler.run([bell], shots=128)
job_result = job.result()
print(f"counts for Bell circuit : {job_result[0].data.meas.get_counts()}")
 
# run a sampler job on the parameterized circuits
job2 = sampler.run([(pqc, theta1), (pqc2, theta2)])
job_result = job2.result()
print(f"counts for parameterized circuit : {job_result[0].data.meas.get_counts()}")

# --------------------------------------------------
# Simulating with noise model from actual hardware
# --------------------------------------------------
from qiskit_ibm_runtime import QiskitRuntimeService
provider = QiskitRuntimeService(channel='ibm_quantum', token="set your own token here")
backend = provider.get_backend("ibm_kyoto")

# create sampler from the actual backend
sampler = SamplerV2.from_backend(backend)

# run a sampler job on the parameterized circuits with noise model of the actual hardware
bell_t = transpile(bell, AerSimulator(basis_gates=["ecr", "id", "rz", "sx"]), optimization_level=0)
job3 = sampler.run([bell_t], shots=128)
job_result = job3.result()
print(f"counts for Bell circuit w/noise: {job_result[0].data.meas.get_counts()}")
```

## Contribution Guidelines

If you'd like to contribute to Aer, please take a look at our
[contribution guidelines](CONTRIBUTING.md). This project adheres to Qiskit's [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-aer/issues) for tracking requests and bugs. Please use our [slack](https://qiskit.slack.com) for discussion and simple questions. To join our Slack community use the [link](https://qiskit.slack.com/join/shared_invite/zt-fybmq791-hYRopcSH6YetxycNPXgv~A#/). For questions that are more suited for a forum, we use the Qiskit tag in the [Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit).

## Next Steps

Now you're set up and ready to check out some of the other examples from the [Aer documentation](https://qiskit.github.io/qiskit-aer/).

## Authors and Citation

Aer is the work of [many people](https://github.com/Qiskit/qiskit-aer/graphs/contributors) who contribute to the project at different levels.
If you use Qiskit, please cite as per the included [BibTeX file](https://github.com/Qiskit/qiskit/blob/main/CITATION.bib).

## License

[Apache License 2.0](LICENSE.txt)
