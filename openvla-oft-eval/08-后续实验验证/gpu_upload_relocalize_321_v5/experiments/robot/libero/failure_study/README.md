# LIBERO-Spatial failure-study package

This directory contains the runnable package for the five targeted failure states.

Start with [`OPERATION_MANUAL.md`](OPERATION_MANUAL.md). It defines the state mapping,
safe Miniconda activation, experiment order, acceptance criteria, and the boundary
between simulator-only diagnostics and deployable policy changes.

Entry points:

- `run_inference_study.sh`: 8-step control, first-4 execution, camera dumps, wrist-feature weighting.
- `run_simulation_study.sh`: action-boundary logs, target-motion replanning, 20/50 Hz comparison.
- `run_modelscope_233_378_analysis.sh`: ModelScope GPU entry point for the 233/378 boundary and grasp-recovery study.
- `verify_oft_step.sh`: run Python, GPU, evaluation-import, checkpoint, and model-load checks one stage at a time.
- `run_top_drawer_sweep.sh`: 20--50 top-drawer initial-state trials and a DAgger collection manifest.
- `run_dagger_finetune.sh`: LoRA fine-tuning after expert corrections have been converted to RLDS.
- `analyze_action_chunks.py`: produces episode/chunk CSV files and a Markdown diagnostic summary.
- `summarize_targeted_eval.py`: produces per-state outcomes and the DAgger collection manifest.
- `CONCLUSION_TEMPLATE.md`: fill this after the runs; do not infer missing measurements.

The scripts intentionally do not activate Conda themselves. Activate the `oft`
environment in the calling terminal so an activation failure cannot close the terminal.
