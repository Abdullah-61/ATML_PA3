# ATML PA3

Code and results for three experiments. All runs used seed 6304 on free Kaggle T4 GPUs.
Raw datasets and checkpoints are not included. Everything downloads at run time.

| Folder | Task | Contents |
|---|---|---|
| `task1_llm/` | LLM fine-tuning (Full FT, L2-SP, LoRA-4/16, OPLoRA) | training, lm-eval, diagnostics, qualitative prompts, `results/` |
| `task2_vlm/` | SigLIP + Qwen3 VLM and grounding tests | data prep, 3-stage training, final eval, `results/` (incl. selected IDs) |
| `task3_ttc/` | Test-time compute | offline selection and adaptive stopping, live generation, `results/` |

## Reproduce

```bash
pip install -r requirements.txt
```

**Task 1** (resumable; rerun the same command after a disconnect)
```bash
cd task1_llm
python run_part.py --root OUT --steps train:full eval:full_mid eval:full train:l2sp eval:l2sp \
                                      train:lora4 eval:lora4 train:lora16 eval:lora16 train:oplora eval:oplora eval:baseline
python diagnostics.py   # weight/activation diagnostics and qualitative generation (needs the trained outputs)
```

**Task 2**
```bash
cd task2_vlm
python prep.py --data DATA select && python prep.py --data DATA fetch && python prep.py --data DATA cache
python train.py --data DATA --out OUT/run196 --pool 1 && python final_eval.py --data DATA --out OUT/run196 --pool 1
python train.py --data DATA --out OUT/run49  --pool 2 && python final_eval.py --data DATA --out OUT/run49  --pool 2
```

**Task 3** (needs the supplied ATML-PA3-Data rollouts and MathVista `testmini.json`)
```bash
cd task3_ttc
python offline.py --data ATML-PA3-Data --meta testmini.json --out OUT
python live.py --data ATML-PA3-Data --out OUT --shard 0 --n_shards 2   # one process per GPU
python live.py --grade OUT
```

## Notes
- Result files are in each task's `results/` folder. Subfolder paths are joined with underscores, for example `evals_full_boolq.json` and `run196_final_vqa_test_correct.jsonl`. Split and example IDs: `task2_vlm/results/data_selected_ids.json`, `task3_ttc/results/splits.json`, `task1_llm/probe_set.json`.
- `diagnostics.py` expects Kaggle paths. Edit the `/kaggle/input` globs for a local run.
- Task 1 qualitative prompts: `first_plot_questions.yaml` from the Emergent Misalignment repository.
- Reused external code and tools: Hugging Face `transformers` and `datasets`, `lm-evaluation-harness`, `math-verify`, MathVista answer-normalization rules. LoRA, OPLoRA and L2-SP are implemented here from the papers.
