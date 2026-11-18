# Data, model, and code provenance

This project uses public English financial numerical questions from FinQA. Its conservative numerical subset and scoring protocol differ from the official full-FinQA benchmark.

## Sources and licenses

| Component | Source and pinned version | License scope |
|---|---|---|
| FinQA data | [Official repository](https://github.com/czyssrs/FinQA), commit `0f16e2867befa6840783e58be38c9efb9229d742`; `dataset/train.json`, `dev.json`, `test.json` | Repository [MIT license](https://github.com/czyssrs/FinQA/blob/0f16e2867befa6840783e58be38c9efb9229d742/LICENSE). No separate dataset license was found during inspection; repository licensing does not establish ownership of third-party reports. |
| FinQA upstream code | Same repository and commit | MIT; this project does not run upstream training code or execute annotated programs. |
| Base model | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), revision `15852e8c16360a2fea060d615a32b45270f8a8fc` | [Apache-2.0](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE); text backbone only. |
| Project code | Root [LICENSE](../LICENSE) | MIT; upstream data, models, and dependencies retain their own licenses. |
| Dependencies | [requirements.txt](../requirements.txt), tested [environment snapshot](../docs/environment.txt) | Individual dependency licenses and notices apply. |

Preserve applicable upstream attribution and notices when redistributing upstream materials or derivatives. Data and model weights are not bundled in Git. See the [FinQA field documentation](https://github.com/czyssrs/FinQA/blob/0f16e2867befa6840783e58be38c9efb9229d742/README.md) and [FinQA paper](https://arxiv.org/abs/2109.00122).

## Download and traceability

Run from the repository root:

```bash
python -m data.prepare_data --download
# Alternatively, supply the official JSON files locally:
python -m data.prepare_data --raw-dir /path/to/official-json
```

Downloads use the pinned commit and record URLs, SHA256 hashes, and counts in `processed/manifest.json`. `--download` refreshes the three cached JSON files. Failures raise errors rather than substituting synthetic data. Offline inputs use `local-unverified:<hash>` because a local file hash alone does not establish the upstream revision.

Original train/dev/public-test splits contain **6251/883/1147** records and are never reshuffled. `private_test.json` lacks labels and is not downloaded or scored. Training entry points reject dev/test records.

## Generated files and model inputs

Official records contain `id`, `pre_text`, `post_text`, `table`, `table_ori`, and `qa`, including `question`, `answer`, `exe_ans`, and `program`.

- `{train,dev,test}.jsonl` retains source fields, revision, split, prompts, verified references, and exclusion reasons.
- `{train,dev,test}_sft.jsonl` contains verified `id/split/messages` records. Only train is optimized. Targets normalize existing annotated answers to JSON, without generated explanations or invented steps.
- `train_grpo.jsonl` contains `id/split/prompt/reference`. Only `prompt` enters the model; reward callbacks receive the reference.

Prompts include complete `pre_text`, all table rows as JSON arrays, and complete `post_text`, in order. Answers, annotated programs, gold evidence, and retrieval outputs are excluded. Training skips whole overlength records and logs IDs. Evaluation keeps them as `context_too_long`; scorable records remain in the accuracy denominator.

## Numerical conventions and conservative filtering

Outputs are JSON such as `{"answer":20,"unit":"percent"}` or `{"answer":0.2,"unit":"ratio"}`. `percent` and `%` convert to dimensionless fractions; `ratio` is already a fraction. Explicit USD scales convert to base dollars; counts and years are separate dimensions.

The parser supports signs, scientific notation, valid thousands separators, parenthesized negatives, and Unicode minus. It rejects unknown/missing units, booleans, NaN/Infinity, duplicate keys, invalid numbers, and surrounding prose. Units and scales are not guessed from numerical proximity.

Reference construction requires an explicit percent sign, supported unit suffix, or a question explicitly asking for a ratio. Bare numbers, unspecified dollar currencies, and isolated scales such as "million" are excluded. Written answers must agree with `exe_ans` within half a unit of the final written digit. Only explicit percent annotations permit reconciling fraction/percentage-point conventions. Conflicting signs, execution values, and non-equivalent plausible scales are not silently repaired.

| Split | All records | Verified references |
|---|---:|---:|
| Train | 6251 | 3428 |
| Dev | 883 | 481 |
| Public test | 1147 | 638 |

All currently retained references are ratios/percentages. Other units have parser tests but no retained examples in this snapshot. Scores describe this restricted subtask, not all FinQA questions. SFT further retained 3363 train records under its token budget.

`reference.value` is the verified execution value; `reference.written_value` provides the normalized annotated SFT target. `reference.atol` captures rounding. Correctness requires the same unit dimension and error within `max(config.atol, reference.atol, config.rtol * abs(reference.value))`. Final accuracy excludes format reward. Test labels are not used to tune parsing, tolerances, rewards, or checkpoint selection.

## Duplicate checks and limitations

`duplicates.json` checks IDs, whitespace/case-normalized full context-plus-question pairs, question-only matches, and report IDs. Duplicate IDs or identical cross-split context/question pairs stop preparation. Question-only and report matches are recorded without changing official splits.

The pinned snapshot has no duplicate IDs or identical cross-split context/question pairs, 73 within-split duplicate context/question groups, 25 cross-split question-only groups, and no cross-split report-ID overlap. Original records are retained, so within-split duplicates may receive more training weight. Checks do not detect paraphrases, equivalent tables, or exposure during model pretraining.

Temporary test examples are explicit program fixtures, not real FinQA data or model predictions.
