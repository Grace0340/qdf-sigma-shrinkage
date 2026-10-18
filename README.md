# Code and logs for "What Does a Meta-Learned Horizon Weighting Learn? A Shrinkage View of Quadratic Direct Forecasting"

Code, run records and frozen protocols of the letter, submitted to
IEEE Signal Processing Letters by Yao Guo and Penggui Peng.

The extended results cited in the letter as Tables E1–E12 are not part of this repository; most of their table rows are among the files in `paper/generated/`.

## What is here

| Path | Content |
|---|---|
| `experiments/q1a_qdf_attribution/protocol*.json` | Frozen protocols, byte-identical to the versions frozen before each protocol's runs. Their SHA-256 prefixes are listed on the Information for Reproducibility page. |
| `experiments/q1a_qdf_attribution/run_grid*.py` | Runner of each protocol. |
| `experiments/q1a_qdf_attribution/summarize*.py` | Frozen decision rules of each protocol. They read only the run records and write `results/summary*.json`. |
| `experiments/q1a_qdf_attribution/cpu_launch.py` | Launcher. It runs the unmodified QDF or TQNet entry point, stubs optional packages that the runs do not use, and imports the patch below when `Q1_SIGMA_MODE` is set. |
| `experiments/q1a_qdf_attribution/fixed_sigma_patch.py` | Replaces the learned Σ by a fixed target after the meta phase. The targets are I + aS0n, the doubled-diagonal form, the replay, the label covariance, random eigenvectors, Ledoit–Wolf and the learned Σ. The meta phase still runs with meta learning rate 0, so initialization, random-number use and batch order match the QDF run with the same seed. |
| `experiments/q1a_qdf_attribution/extras_v8.py`, `analyze_sigma_shape.py` | Shape analysis of the learned matrices (post hoc, descriptive). |
| `experiments/q1a_qdf_attribution/results/` | Every run record (`runs*.jsonl`), every summary, and the runner logs. |
| `experiments/q1a_qdf_attribution/v8_pulled/runs_v8/` | Per-run logs and configurations of protocol v8. Matrix files are kept only for the arms the number scripts read: QDF_LR1 `A.pth`, and REPLAY_K1 and PLUGIN_A1 `sigma_target.npy`. |
| `experiments/q1a_qdf_attribution/v9_pulled/results/` | Run records of protocol v9 (fresh seeds 2026–2028, `runs_v9.jsonl`), the 16 repeats of v8 runs on the v9 machine (`runs_v9_xm.jsonl`), the frozen summary, the runner logs and the library versions of the v9 environment. |
| `experiments/q1a_qdf_attribution/v9_pulled/runs_v9/` | Per-run logs, configurations and metrics of protocol v9. No matrix files. |
| `paper/scripts/v8_numbers.py` | Recomputes every per-seed gain and validation choice of v8 and checks them against `summary_v8.json`. It writes the v8 part of Table I, the data of Fig. 1 and the v8 numbers of the text to `paper/generated/`. |
| `paper/scripts/v9_numbers.py` | Recomputes every per-seed gain and validation choice of v9, checks them against `summary_v9.json` and checks that the 16 repeats equal the v8 runs. It writes Table I, Tables E10 and E11 and the v9 numbers of the text, including the post hoc summaries, which are marked as such in the letter. Run it after `v8_numbers.py`. |
| `paper/scripts/gen_numbers.ps1`, `stats_checks.py` | Numbers, intervals and table rows of the earlier protocols, from the summaries. |
| `paper/scripts/resim_meta.py` | Replay of the released meta phase on a constant moment, with no forecaster. |
| `paper/derivations/qdf_meta_gradient_check.py`, `paper/scripts/toy_checks_flow.py` | Numerical checks of Propositions 1–3 on a toy problem (Table E1). They do not import QDF. |
| `paper/generated/` | Output of the number scripts, exactly as used to compile the letter. |
| `SHA256SUMS.txt` | Checksum of every file in the package. |

The 96 matrix files (`.npy`, `.pth`) are needed only for the shape errors of Section III-B; every other number of the letter needs only the text files.

Not included:

- the data sets, which are public;
- the QDF and TQNet code;
- model checkpoints and the learned matrices of the other arms (about 1.4 GB for v8), which are available from the authors.

## Edits made for this copy

1. Absolute local paths in scripts and run records were replaced by `<HOME>` or `<PROJECT>`, and host names in run records by `<host>`. Protocol files were not edited.
2. In `v8_pulled/runs_v8/<run>/res/`, `v9_pulled/runs_v9/<run>/res/` and `ck/`, the long result-folder name written by QDF was shortened to `run/`, to stay within the Windows path limit. The number scripts locate this folder by wildcard.
3. Nothing else was changed.

## Setup

1. Put the upstream code next to the runners. Both are used unmodified:
   ```
   cd experiments/q1a_qdf_attribution
   git clone https://github.com/Master-PLC/QDF && git -C QDF checkout eb0693a962928e229417fd80b401c37b0dac6a67
   git clone https://github.com/ACAT-SCUT/TQNet && git -C TQNet checkout 15e19cb23483ed52398566c4baa959168cfffa57
   ```
2. Put the data under `QDF/dataset/`:
   - the ETT files go in `ETT-small/`;
   - `weather.csv` and `electricity.csv` are the standard Autoformer files and go in the folders named in `run_grid_v5.py`.

   The SHA-256 prefixes of all data files are in `protocol_v8.json` under `platform`.
3. Python 3.12 with PyTorch 2.13.0, numpy 1.26.4, pandas 2.3.3, numba 0.67.0 and scipy. Set `Q1_PY` to the interpreter if it is not the one running the scripts.

## Reproducing the letter

The paper numbers need only the files in this package. They do no training.

```
python paper/scripts/v8_numbers.py
python paper/scripts/v9_numbers.py
python paper/scripts/stats_checks.py
powershell -ExecutionPolicy Bypass -File paper/scripts/gen_numbers.ps1
python paper/derivations/qdf_meta_gradient_check.py
python paper/scripts/toy_checks_flow.py
```

Rerunning `v8_numbers.py` and `v9_numbers.py` in this package reproduces every file in `paper/generated/` byte for byte, except `v8.json`, whose matrix norms can differ in the 16th significant digit (floating-point summation order).

Rerunning the experiments needs the setup above and a GPU. Each command resumes after an interruption, and therefore skips every run already recorded in `results/`: move the shipped `results/` folder aside before a fresh rerun.

```
cd experiments/q1a_qdf_attribution
python run_grid_v8.py --jobs 10
python summarize_v8.py
python run_grid_v9.py --xm      # 16 repeats of v8 runs, to check the platform
python run_grid_v9.py --jobs 12
python summarize_v9.py
```

A rerun of v9 writes to `results/runs_v9.jsonl`; the shipped v9 records, which `v9_numbers.py` reads, are in `v9_pulled/results/`.

The earlier protocols follow the same pattern: v5 has `--stage A` and `--stage B`; v1–v4 ran on CPU.

A v8 run took 103–1534 s on an NVIDIA RTX 4080 SUPER, with several jobs sharing the GPU. The run times of the 778 runs sum to 107 h. A v9 run took 67–1269 s on a second machine of the same type; the 590 runs sum to 61 h.

## Upstream licences

QDF is distributed under the MIT licence and TQNet under the Apache 2.0 licence. Neither is redistributed here.
