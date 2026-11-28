# Quadratic APN functions on F_2^8: code and data

Code and small data files of the paper

> E. Bellini, *How many quadratic APN functions are there in dimension eight?*, 2026.

The paper gives a uniform sampler. It draws every APN bilinear map on F_2^8, that is every quadratic APN
(8,8)-function up to affine terms, with the same probability. It builds on the variable approach of Beierle,
Langevin, Leander, Polujan and Rasoolzadeh (arXiv:2508.04644, "BLLPR"), and runs on CPUs and NVIDIA GPUs. It found
2,000,780 classes that are in no published list. A class invariant proves that the 5,813,648 known classes are
pairwise inequivalent. In a model of independent draws, it is very unlikely that a value of the invariant was
missed; if the invariant also separates the classes, 5,813,648 is very likely the number of classes. The paper
also searches exhaustively, with a computer, the functions with a linear self-equivalence of prime order (every
one has the invariant of one of 12,775 published classes), and reports the methods that did not find classes
faster.

The scripts are copied as they were run, except that `uniform/mass8.c` now reports incomplete searches in its test mode (`-g7`). Their relative paths refer to the original working tree (see "Paths").
The numbers of the paper are in `estimate/numbers.tex`, as written by `estimate/make_numbers.py`.

## Where the class lists are

The class lists are large, so they are kept in the database
[peacker/quadratic-apn-classes-n8-n10](https://github.com/peacker/quadratic-apn-classes-n8-n10) (folder `n8/`):

- `n8/classes/`: every known class, with its class key, M, the result of the permutation test and the works
  that list it (this paper is source `bellini26`);
- `n8/luts/`: one lookup table for each class of `bellini26`, by class key.

The class key is `sha1(repr(ortho_invariant(F)))[:16]` (`lib/apnsearch/invariants.py`), where the invariant is the
differential spectrum and the extended Walsh spectrum of the ortho-derivative.

## Contents

| Folder | Paper | What |
|---|---|---|
| `lib/apnsearch/` | Sec. 2 | Python library: APN test, ortho-derivative invariant, CCZ-to-permutation test (counting criterion and dual test) |
| `uniform/` | Sec. 3, 5 | `mass8.c`: the uniform sampler on CPUs (`-mass`; build with `-DSPLITMIX`); `mass_worker.sh`, `mass_gpu_worker.sh`: one server worker on a CPU thread or on a GPU; `small.c`: exact check of the estimator for n = 4, 5 (Sec. 5.4); `rngtest/`: step-1 acceptance with xorshift64 and splitmix64 (Sec. 3.6; `summary.py` prints it) |
| `gpu/cuda/` | Sec. 3.5 | `mass8g.cu`: the uniform sampler on NVIDIA GPUs (same steps and generator as `mass8 -mass`); `cpu.c`, `Makefile` |
| `sampler/` | Sec. 3, 4, 5.5, 7 | `testall.py`: class key, M and permutation test of each output (every output function is tested); `fastclass.py`: the same with results cached by key, used only to classify the leaves of the symmetric search; `build_db.py`: class index from the outputs; `varext.c`, `varext_worker.sh`: the sampler of the independent sample (Sec. 5.5); `profile.c`: bent-component profile against the filters of BLLPR (Sec. 4.3); `gen6.c`, `switch2.c`: coordinate approach and switching (Sec. 7) |
| `estimate/` | Sec. 4, 5 | `make_numbers.py`: every number of the paper, as LaTeX macros (`numbers.tex`); `rarefaction.py`: the data snapshot that `make_numbers.py` reads; `uniform.py`: Figure 1; `freq_of_freq.tsv`: how many classes outside the published lists were drawn once, twice, ...; `calls.tsv.gz`: one line per included call (starts, attempts, tries, valid L6, output rows, flags, draws outside P and K); `reproduce.py`: recomputes the statistics of Sec. 5.2-5.3 from these files alone and compares them with `numbers.tex` |
| `symmetric/` | Sec. 6 | `types8.tsv`: the 107 types; `sym_setup.py`: space W of each type (`types/W/`); `engine/sym10t.c`: search engine (built as `sym8k`); `engine/autq.c`: order of G(F); `symmetric_group_orders.txt`: its output for the 12,775 classes, used for the exact weights; `engine/typeaudit.py`: audit against published automorphism groups; `engine/fixlemma.py`: fixed-space exclusions; `classify.py`, `enumerate.py`; `types_table.tsv`: every type with its matrices, fixed-space dimensions, exclusion, search run, leaves and classes; `results/`: the outputs of every run and the 12,775 classes (`symmetric_classes_bgp26.tsv`) |
| `pilots/` | Sec. 7 | the attempts that did not find classes faster: restriction starts (`restrict6.py`, `perstart.sh`, `perstart_eval.py`), relaxed filters (`m4/gen6b.c`), M = 4 by construction (`m4/gen4m.c`), structure of the M = 4 classes (`m4/*.py`); `results/`: their outputs and logs, and the profile tables of Sec. 4.3 (`prof_new.tsv`, `prof_pub.tsv`) |
| `langevin/` | Sec. 4.1 | `orthokey.c`, `hashkeys.py`: fast class keys for the 11.9 million functions of P. Langevin's data file (checked against `lib/apnsearch` by `checkpy.py`); `mkshards.py`, `compare.py`: the comparison with the other lists |
| `tools/check_manifest.py` | | SHA-256 of every tracked file (`MANIFEST.sha256`) |

## Requirements

- Python 3.10 or later with NumPy and SciPy (`pip install -r requirements.txt`); matplotlib for the figure.
- A C compiler. `mass8.c` and `varext.c` need AVX2 (x86-64) or NEON (ARM).
- For the GPU version: an NVIDIA GPU, the CUDA toolkit (tested with nvcc 12.1) and gcc.

## Running the uniform sampler (CPU or GPU)

Both versions make the same draws and print the same output: one line per solution L7 (two per positive G7), and a
counter line on stderr (`starts chain_attempts L6_tries valid_L6 outputs nonstandard tag`). The tag is `splitmix2`
on the CPU and `splitmix2g` on the GPU. `testall.py` then classifies the outputs.

CPU (any x86-64 or ARM machine):
```
gcc -O2 -march=native -DSPLITMIX -o mass8 uniform/mass8.c
./mass8 -mass 30000 -seed 1 -tries 4096 -cap 64 > out.txt 2> log.txt     # 30,000 starts, about 13,000 G7
python3 sampler/testall.py --n 8 --source cpu_1 --input out.txt --out rows.tsv --keep-lut
```

GPU (`make` builds both `mass8g` and the CPU `mass8`; use `ARCH=sm_80` for an A100):
```
cd gpu/cuda && make ARCH=sm_75
./mass8 -mass 10000 -seed 7 -tries 4096 -dump g7.bin > /dev/null 2>&1    # reference G7 from the CPU
./mass8g -check g7.bin -dev 0                                             # must report 0 mismatches
./mass8g -mass 6000000 -seed 1 -dev 0 > out.txt 2> log.txt                # about 2.6M G7, 2-3 minutes
cd ../.. && python3 sampler/testall.py --n 8 --source gpu_1 --input gpu/cuda/out.txt --out rows.tsv --keep-lut
```

Use a different seed for every call. On servers, `uniform/mass_worker.sh` (CPU) and `uniform/mass_gpu_worker.sh` (one
per GPU) loop over calls, classify each call and write one counter line per call; `estimate/make_numbers.py` reads
both kinds.

## Checks

| Check | Paper | How |
|---|---|---|
| GPU and CPU find the same solutions | Sec. 3.5 | `mass8 -dump` writes the G7 of a CPU run; `mass8g -check` searches them on the GPU and reports mismatches. On 60,273 G7 (14 seeds): 0 mismatches. |
| GPU and CPU have the same rates | Sec. 3.5 | One GPU call `-mass 600000 -seed 12345`: valid L6 per try 1.055e-4 (CPU 1.051e-4), chain attempts per start 14.95 (14.94), outputs per G7 1.064e-3 (1.062e-3). |
| The estimator is exact in small dimension | Sec. 5.4 | `small.c` enumerates all APN bilinear maps for n = 4, 5 and compares with the estimate. |
| The generator matters | Sec. 3.6 | `uniform/rngtest/summary.py` prints the step-1 acceptance rates of the runs in `rngtest/`. |
| The symmetric search agrees with the literature | Sec. 6.4 | `symmetric/engine/typeaudit.py` and `autq.c` compare the classes found for each type with the published automorphism groups. |
| Every output is APN, with the right key and test | Sec. 3.4 | `sampler/testall.py` recomputes everything from the lookup table. |
| The statistics follow from the released data | Sec. 5 | `python3 estimate/reproduce.py` (about 10 s) recomputes the counts, the count of bilinear maps, the dispersion and the completeness estimate from `calls.tsv.gz`, `freq_of_freq.tsv` and `symmetric/symmetric_group_orders.txt`. |
| No search was cut short | Sec. 3.2 | No line of `calls.tsv.gz` has more than 46 output rows on a CPU, while one G7 at the limit of 64 orbits gives 128; GPU calls are marked if the uncapped GPU count differs from the output, and no included call is marked. `mass8 -g7` exits with status 2 or 3 if its search was incomplete. |

## Environment of the runs

- Uniform sampler, CPU: Linux servers (Docker containers, Ubuntu 22.04) with 2 x AMD EPYC 7763, shared with other
  users, at `nice 19`, at most 200 threads per server, and the CPUs of the GPU servers (Xeon Gold 5222, EPYC 7742).
  Calls with xorshift64 (21-22 November 2026) are reported separately; all counts use splitmix64.
- Uniform sampler, GPU: 32 GPUs (RTX 2080 Ti, Quadro RTX 8000, TITAN RTX) in four servers, and 4 A100 in a fifth
  server.
- Independent sample (`varext`): the EPYC 7763 servers, October to November 2026.
- Symmetric search and pilots: a laptop (Apple M5 Pro, 15 cores) and the same servers.

## Paths

The scripts were run in a working tree with this layout. The relative paths inside the scripts refer to it.

| Working tree | Here |
|---|---|
| `learning/lib/apnsearch` | `lib/apnsearch` |
| `learning/quad/` | `sampler/` |
| `learning/quad/server_results/*/results/varext/w*.tsv` | raw output of both samplers, one row per function (not included: about 13 GB; every class is in the database) |
| `learning/quad/db/` | class index (`build_db.py`) |
| `learning/sym10/`, `learning/sym10/s8/` | `symmetric/`, `symmetric/engine/` |
| `n8search/pilots/` | `pilots/` |
| `papers/n8-quadratic/tools`, `figures` | `estimate/` |
| `n8search/mass/`, `n8search/mass/deploy/` | `uniform/` |
| `n8search/cuda/` | `gpu/cuda/` |

## License

MIT (see `LICENSE`).
