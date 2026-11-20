import json
import math
from types import MethodType

import pytest

from rewards.dapo import outcome


def test_outcome_units_and_soft_length_penalty():
    ref = {'value': .2, 'unit': 'ratio'}
    args = (128, 32, 1.0, {'atol': 1e-6, 'rtol': .001})
    assert outcome('{"answer":20,"unit":"percent"}', ref, 20, *args)['reward'] == 1
    assert outcome('{"answer":0,"unit":"ratio"}', ref, 20, *args)['reward'] == -1
    assert outcome('{}', ref, 20, *args)['reward'] == -1
    assert outcome('{"answer":.2,"unit":"ratio"}', ref, 112, *args)['length_penalty'] == -.5
    assert outcome('{}', ref, 128, *args)['reward'] == -2


def test_complete_group_filtering_and_truncation():
    pytest.importorskip('trl')
    from train.dapo_trainer import filter_groups

    def row(key, correct, terminated=True):
        return {'example': {'id': key}, 'score': {'correct': correct}, 'terminated': terminated}

    # Rank boundaries may split a group; filtering sees rank-ordered GLOBAL rows.
    rows = [row('easy', True)] * 4 + [row('mixed', True)] * 2 + [row('mixed', False)] * 2
    kept, diagnostics = filter_groups(rows, 4)
    assert len(kept) == 4 and {r['example']['id'] for r in kept} == {'mixed'}
    assert [r['accepted'] for r in diagnostics] == [False, True]
    assert not filter_groups([row('truncated', True), row('truncated', False, False)], 2)[0]
    with pytest.raises(ValueError, match='Incomplete'):
        filter_groups(rows[:-1], 4)
    with pytest.raises(ValueError, match='IDs'):
        filter_groups([row('a', True), row('b', False)], 2)


def test_dynamic_refill_native_loss_and_best_export(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('trl')
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from safetensors.torch import load_file
    from test_validation import tiny_model_and_tokenizer, dev_row
    from rewards.dapo import DAPOReward
    from train.dapo_trainer import NumericalDAPOTrainer
    from financeqa.distributed import save_training
    import time

    model, tok = tiny_model_and_tokenizer()
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    rows = [{'id': f'train-{i}', 'prompt': [{'role': 'user', 'content': f'question {i}'}],
             'reference': {'value': .2 if i % 2 else 0, 'unit': 'ratio'}} for i in range(6)]
    references = {r['prompt'][0]['content']: r['reference']['value'] for r in rows}
    calls = []

    def controlled_rollout(self, prompts):
        calls.append(prompts)
        pids, cids = [], []
        for i, prompt in enumerate(prompts):
            ref = references[prompt[0]['content']]
            good, bad = (3, 4) if ref == .2 else (4, 3)
            token = good if len(calls) == 1 or i % 2 == 0 else bad
            pids.append([5, 6 if ref == .2 else 7])
            cids.append([token, 1])
        texts = [[{'role': 'assistant', 'content': tok.decode(ids, skip_special_tokens=True)}] for ids in cids]
        return pids, cids, None, texts, None, None, {}, None, []

    monkeypatch.setattr(GRPOTrainer, '_generate', controlled_rollout)

    def dev_generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, torch.tensor([[3, 1]], device=input_ids.device)], dim=1)

    model.generate = MethodType(dev_generate, model)
    out = tmp_path / 'dapo'
    args = GRPOConfig(output_dir=str(out), use_cpu=True, bf16=False, report_to='none',
        max_steps=2, learning_rate=.01, gradient_checkpointing=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_generations=2, num_generations_eval=1, num_iterations=2, max_completion_length=8,
        beta=0, epsilon=.2, epsilon_high=.28, loss_type='dapo', scale_rewards='group',
        eval_strategy='steps', eval_steps=2, save_steps=2, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='numerical_accuracy', greater_is_better=True,
        logging_steps=1, disable_tqdm=True)
    reward = DAPOReward(out / 'reward_groups.jsonl', {'atol': 1e-6, 'rtol': .001}, 2, 8, 2)
    trainer = NumericalDAPOTrainer(model=model, args=args, processing_class=tok,
        train_dataset=Dataset.from_list(rows), eval_dataset=[dev_row()], reward_funcs=[reward],
        validation_config={'data': str(tmp_path/'dev.jsonl'),
            'generation': {'max_input_tokens': 16, 'max_new_tokens': 8, 'do_sample': False},
            'tolerance': {'atol': 1e-6, 'rtol': .001}},
        model_spec={'name': 'tiny-random-fixture', 'revision': 'fixture',
                    'chat_template_kwargs': {'enable_thinking': False}},
        dynamic_sampling={'max_generation_batches': 4}, dapo_reward=reward)
    (tmp_path/'dev.jsonl').write_text(json.dumps(dev_row()) + '\n')
    result = trainer.train()
    save_training(trainer, tok, out, result, time.perf_counter())
    trainer.write_selection()
    # First all-correct batch is rejected; fresh prompts refill it. The second
    # optimization pass reuses precisely the selected behavior-policy rollouts.
    assert len(calls) == 2
    assert {p[0]['content'] for p in calls[0]}.isdisjoint(p[0]['content'] for p in calls[1])
    diagnostic = json.loads((out/'dynamic_sampling.jsonl').read_text())
    assert diagnostic['generation_batches'] == 2 and diagnostic['selected_groups'] == 2
    assert len(set(diagnostic['selected_ids'])) == 2
    groups = [json.loads(l) for l in (out/'reward_groups.jsonl').read_text().splitlines()]
    assert len(groups) == 2 and all(g['accuracy'] == .5 and g['std'] == 1 for g in groups)
    logs = [r for r in trainer.state.log_history if 'loss' in r]
    assert len(logs) == 2 and all(math.isfinite(r['loss']) and r['grad_norm'] > 0 for r in logs)
    assert any(not torch.equal(initial[n], p) for n, p in model.named_parameters() if n in initial)
    assert all(b['old_per_token_logps'] is not None for b in trainer._buffered_inputs)
    assert not any('.ref.' in n for n, _ in model.named_parameters())
    selection = json.loads((out/'best_checkpoint.json').read_text())
    assert selection['export_contains_best']
    exported = load_file(str(out/'adapter_model.safetensors'))
    best = load_file(str(out/'checkpoint-2/adapter_model.safetensors'))
    assert all(torch.equal(v, best[k]) for k, v in exported.items())
