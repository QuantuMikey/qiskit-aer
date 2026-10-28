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

This fork was built to run large tensor network simulations on [LUMI](https://www.lumi-supercomputer.eu/),
whose GPU partition (LUMI-G) uses AMD Instinct MI250X GPUs. It is also
intended for other AMD-based systems, such as LUMI-AI.

### What is different from upstream

| Area | Upstream qiskit-aer | This fork |
|---|---|---|
| Tensor network on AMD GPUs | Not available | GPU contractor on ROCm: each pairwise contraction runs as one hipBLAS ZGEMM (CGEMM in single precision), with the fork's own kernels handling operand layout |
| Contraction path search | Done inside cuTensorNet | [cotengra](https://github.com/jcmgray/cotengra) hyper-optimizer, with optional mt-kahypar and cotengrust |
| Large networks | Slicing handled inside cuTensorNet | Memory-aware slicing, with slices distributed across MPI ranks |
| Path search under MPI | Handled inside cuTensorNet | Search trials sharded across ranks; best plan (lowest FLOP count after slicing) chosen collectively |
| Plan reuse | None | Contraction plans saved to disk and replayed in later runs |
| CPU-only path search | n/a | Search on CPU nodes, contraction later on GPU nodes |
| GPU capability query | n/a | `AerSimulator.gpu_device_count()` and `AerSimulator.tensor_network_gpu_available()` |
| Statevector on AMD GPUs | 32-bit grid dimension overflow at large qubit counts | Fixed in `ExecuteSum`/`ExecuteSum2` and blocked-gate kernels |

Note: the build flag and macro are still named `AER_ENABLE_HIPTENSOR` /
`AER_HIPTENSOR` for historical reasons. The contraction backend is hipBLAS;
hipTensor is no longer linked.

### Split CPU/GPU workflow

For large networks, the contraction path search can take far longer than the
contraction itself, and it does not need a GPU. The fork separates the two:

1. **Search on CPU nodes.** Run the hyper-optimized path search on CPU-only
   nodes (for example LUMI-C), one MPI rank with 16 physical cores. The search
   has two stages: the Mt-KaHyPar partitioning trials, which use all 16
   threads, and cotengra's slicing/reconfiguration, which is single-threaded
   Python and takes about half the search time at p = 9. More threads do not
   help: at 128 threads the partitioning ran 1.6× slower than at 16, and the
   slicing stage is unchanged. Each search gives a different result, so run it
   several times with different seeds. Plans are saved, and the best one
   (lowest FLOP count after slicing) is kept.
2. **Contract on GPU nodes.** Run the contraction on LUMI-G, replaying the
   best saved plan instead of searching again.

The main settings:

| Variable | Purpose |
|---|---|
| `AER_TN_FORGE_ONLY=1` | Run only the path search and save the plan, then exit before any GPU work. Allows `tensor_network` on nodes with no GPU. Pair with `AER_TN_PLAN_FORGE=1`. |
| `AER_TN_PLAN_DIR` | Directory of saved plans (one per network topology). Keeps the best plan per topology. |
| `AER_TN_PLAN_FILE` | Single saved plan file (one topology). |
| `AER_TN_PATH_MAX_REPEATS` | Number of search trials. Under MPI this is the total across all ranks. |
| `AER_TN_PATH_METHODS` | Restrict cotengra's search methods (e.g. `greedy`, `mt-kahypar`). |
| `AER_TN_MTKAHYPAR_THREADS` | Threads for Mt-KaHyPar trials. 16 is the measured sweet spot on LUMI-C. |
| `AER_TN_SLICE_TARGET_BYTES` | Per-slice memory target; lower values slice more aggressively. On a 64 GB MI250X GCD, 8 GiB (`8589934592`) is the measured safe value: the device pool is a liveness layout of all intermediates, about 3× the per-slice peak, so a plan whose peak actually reaches a 16 GiB target does not fit. |
| `AER_TN_MIN_SLICES_PER_RANK` | Minimum slices per MPI rank, so every rank gets work. |
| `AER_TN_HOIST` | Slice-invariant hoisting: steps that do not depend on the sliced indices are computed once and skipped on every later slice. On by default; set `0` when the pinned frontier does not fit device memory (required at p = 9, see below). |

Further tuning and diagnostic variables (`AER_TN_*`) are documented in
comments in `src/simulators/tensor_network/path_optimizer.hpp`.

### Building

Requires ROCm with hipBLAS, a C++17 compiler, and `cotengra>=0.7.0` installed
in the build's Python environment. Enable the tensor network backend in a
ROCm build with:

```bash
export AER_ENABLE_HIPTENSOR=1
```

Then follow the ROCm build instructions in the
[contributing guide](CONTRIBUTING.md). The build stops with an error if
cotengra or hipBLAS is not found.

Usage is the same as upstream:

```python
from qiskit_aer import AerSimulator

sim = AerSimulator(method="tensor_network", device="GPU")
```

### Example: QAOA at p = 9 on LUMI

This example reproduces one row of Table I of
[arXiv:2107.00677](https://arxiv.org/abs/2107.00677): fixed-angle QAOA for
MaxCut at depth p on the depth-p tree subgraph of a 3-regular graph. At p = 9
the circuit has 2046 qubits. The quantity computed is ⟨Z₀Z₁⟩ on the root edge,
from which the cut fraction is C_p = (1 − ⟨Z₀Z₁⟩) / 2.

The run has two steps: a CPU-only path search that saves plans, then a GPU
contraction that replays the best plan. Both steps run the same Python script.

#### Python script (`tree_row.py`)

The fixed angles come from
[danlkv/fixed-angle-QAOA](https://github.com/danlkv/fixed-angle-QAOA)
(3-regular graphs, key `"3"`), stored as `fixed_angles_3reg.json` in the form
`{"9": {"gamma": [...], "beta": [...], "AR": ...}, ...}`.

```python
import json, os, sys

# Import qiskit_aer before mpi4py: Aer initializes MPI at import time.
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator


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


p = int(os.environ["TREE_P"])
row = json.load(open("fixed_angles_3reg.json"))[str(p)]
n, edges = tree_graph(p)
qc = qaoa_circuit(n, edges, row["gamma"], row["beta"])
qc.save_expectation_value(Pauli("ZZ"), [0, 1], label="zz")

sim = AerSimulator(method="tensor_network", device="GPU")
sim.set_max_qubits(n)  # default cap is the statevector limit, not meaningful for TN
result = sim.run(transpile(qc, sim, optimization_level=0), shots=1).result()

# A forge-only run saves its plan and then returns an unsuccessful result
# by design; it does not contract.
if os.environ.get("AER_TN_FORGE_ONLY") == "1":
    ok = "plan capture complete at setup" in str(result.status)
    print("forge OK" if ok else f"forge FAILED: {result.status}")
    sys.exit(0 if ok else 1)

if int(os.environ.get("SLURM_PROCID", 0)) == 0:
    zz = float(result.data(0)["zz"])
    c_p = (1 - zz) / 2
    print(f"p={p} N={n} ZZ={zz:+.15f} C_p={c_p:.15f} "
          f"table={row['AR']:.15f} diff={abs(c_p - row['AR']):.3e}")
```

#### Step 1: path search on a CPU node (`forge.sh`)

Submit this several times with different seeds. Each job adds a plan to the
plan directory; plan file names record the seed, slice target, and FLOP count.

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

# SINGULARITYENV_* variables are passed into the container without the prefix.
export SINGULARITYENV_TREE_P="$P"
export SINGULARITYENV_AER_TN_PLAN_FORGE=1
export SINGULARITYENV_AER_TN_FORGE_ONLY=1
export SINGULARITYENV_AER_TN_PLAN_DIR="$PLANS"
export SINGULARITYENV_AER_TN_MIN_SLICES_PER_RANK=64
export SINGULARITYENV_AER_TN_FORGE_BUDGET_MB=65226   # free memory of one MI250X GCD
export SINGULARITYENV_AER_TN_SLICE_RECONF_EVERY=1
export SINGULARITYENV_MALLOC_ARENA_MAX=2

srun --ntasks=1 singularity exec "$SIF" python3 -u tree_row.py
```

Command used for p = 9 (LUMI-C `small` partition, 16 physical cores;
`--hint=nomultithread` keeps SMT siblings off the allocation):

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

For p = 9 the search was run 32 times at the 8 GiB slice target (24 distinct
seeds; 8 of them repeated on `standard` nodes with 128 threads as a
thread-count comparison), plus 8 times at a 12 GiB target. Searches took 7 to
10 minutes on 16 cores and 9 to 11 minutes on 128. Plan cost varied by a factor
of 10⁴ across seeds. Step 2 used the best plan: seed 12142 at 8 GiB
(1.1e18 FLOPs, 16,384 slices).

#### Step 2: contraction on GPU nodes (`replay.sh`)

The saved plan works at any number of ranks; slices are distributed across
the ranks.

```bash
#!/bin/bash
#SBATCH --job-name=replay
#SBATCH --account=<project>
#SBATCH --partition=standard-g
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7

P="$1"
PLAN="$2"
SIF=<container>.sif
WRAP=/appl/local/quantum/qiskit/run-singularity-with-gpu-affinity   # LUMI GPU-affinity wrapper

# GPU-aware Cray MPICH settings for multi-rank runs. Do NOT set these for a
# single-rank job: with one task the LUMI wrapper aborts at MPI init with
# "GPU_SUPPORT_ENABLED is requested, but GTL library is not linked".
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_CACHE_MAX_SIZE=100
export MPICH_GPU_IPC_THRESHOLD=524288
export MPICH_OFI_NIC_POLICY=GPU

# One rank per GCD, bound to the CPU cores closest to it
MASK=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000

export SINGULARITYENV_TREE_P="$P"
export SINGULARITYENV_AER_TN_PLAN_FILE="$PLAN"
export SINGULARITYENV_AER_TN_GPU_PERSIST=1
export SINGULARITYENV_AER_TN_RG_THREADS=7
export SINGULARITYENV_MALLOC_ARENA_MAX=2

srun --cpu-bind=$MASK "$WRAP" "$SIF" python3 -u tree_row.py
```

Command used for p = 9 (256 LUMI-G nodes, 2048 GCDs, about 6 minutes wall time):

```bash
sbatch --partition=standard-g --nodes=256 --ntasks=2048 \
  --gpus-per-task=1 --cpus-per-task=7 --mem=480G --time=00:15:00 \
  --export=ALL,SINGULARITYENV_AER_TN_HOIST=0 \
  replay.sh 9 /abs/path/plans/plan_<key>.s12142.t8g.<...>.ctg
```

`AER_TN_HOIST=0` is required at p = 9: slice-invariant hoisting pins a 27 GB
frontier into the device pool on top of the 28 GiB the plan already needs,
which exceeds a 64 GB GCD. At p ≤ 8 hoisting is on by default and fits.

Use `--ntasks` (not `--ntasks-per-node`) and `--gpus-per-task=1` (not
`--gpus-per-node`) on `standard-g`; otherwise the GPU-affinity wrapper can
hide the GPUs from the ranks. Keep per-step diagnostic logging (`AER_TN_VERBOSE`)
off for large runs: at 2048 ranks it pushed ~25 million lines through srun's
stdio and stretched setup from 30 seconds to over 7 minutes.

#### Results on LUMI-G

| p | Qubits | LUMI-G nodes | GCDs | Slices | Run time | GCD-hours | Diff vs. paper |
|---|---|---|---|---|---|---|---|
| 1 | 6 | 1 | 4 | 16 | 12 s | 0.01 | 4e-8 |
| 2 | 14 | 1 | 4 | 16 | 11 s | 0.01 | 1e-7 |
| 3 | 30 | 1 | 4 | 16 | 11 s | 0.01 | 3e-7 |
| 4 | 62 | 1 | 4 | 16 | 12 s | 0.01 | 7e-7 |
| 5 | 126 | 1 | 4 | 16 | 12 s | 0.01 | 1e-6 |
| 6 | 254 | 1 | 4 | 128 | 16 s | 0.02 | 3e-6 |
| 7 | 510 | 1 | 4 | 16 | 20 s | 0.02 | 1e-5 |
| 8 | 1022 | 1 | 8 | 64 | 34 s | 0.08 | 2e-5 |
| 9 | 2046 | 256 | 2048 | 16,384 | 325 s | 185 | 5e-5 |

Each AMD MI250X has two GCDs, and each GCD appears to the software as one
GPU. "Run time" is the simulator's own timer (transpile, contraction and
reductions); container start is excluded. The p = 9 job was billed as 6:13 of
wall time on 256 nodes, about 212 GPU-hours. "Diff vs. paper" is the absolute
difference between the computed C_p and the published value. The published
values carry their own optimizer residual, so differences below 1e-4 are
treated as a match. For p ≤ 5 and p = 7 the 16 slices are the forge's
distribution floor, not a memory requirement; from p = 6 the slice count is set
by the memory envelope. The p = 9 path search is not included in the
GCD-hours; it ran on CPU nodes.

### Status

Experimental. Results have been validated by reproducing Table I of
[arXiv:2107.00677](https://arxiv.org/abs/2107.00677) up to p = 9 (QAOA depth;
2046 qubits). Larger p was not attempted: cost grew well over 1000× from p = 8
to p = 9, and the p = 9 run was already at the memory limit of a 64 GB GCD
(hoisting had to be disabled to fit).

Related: [Quantum simulations on LUMI with Qiskit Aer](https://fiqci.fi/publications/2025-04-01-LUMI-quantum-simulations-qiskit-aer) (FiQCI blog).

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
