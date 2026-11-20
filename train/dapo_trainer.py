"""DAPO dynamic sampling on pinned TRL 1.17.1, before native tensor/loss preparation.

Only complete, correctly aligned text groups enter training. Rejected candidates
never enter the loss. All ranks select the same global groups, then repartition
them together with their references, token IDs, and decoded completions.
"""
import json
from pathlib import Path
import random

from transformers import TrainerCallback

from financeqa.distributed import gather_records
from train.validation import NumericalGRPOTrainer


class DAPOProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            path = Path(args.output_dir) / 'live_metrics.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'step': state.global_step, 'max_steps': state.max_steps,
                                              'epoch': state.epoch, **(logs or {})}, allow_nan=False) + '\n')
            temporary.replace(path)


def filter_groups(records, group_size):
    if len(records) % group_size:
        raise ValueError('Incomplete DAPO generation groups')
    accepted, diagnostics = [], []
    for start in range(0, len(records), group_size):
        group = records[start:start + group_size]
        if len({r['example']['id'] for r in group}) != 1:
            raise ValueError('DAPO group IDs disagree across ranks')
        correctness = [bool(r['score']['correct']) for r in group]
        # Masked/truncated rows cannot be the sole source of either outcome.
        usable = {bool(r['score']['correct']) for r in group if r['terminated']}
        keep = usable == {False, True}
        diagnostics.append({'id': group[0]['example']['id'], 'n': len(group),
                            'correct': sum(correctness), 'accepted': keep,
                            'truncated': sum(not r['terminated'] for r in group)})
        if keep:
            accepted.extend(group)
    return accepted, diagnostics


class NumericalDAPOTrainer(NumericalGRPOTrainer):
    def __init__(self, *args, dynamic_sampling, dapo_reward, **kwargs):
        self.dynamic_sampling = dynamic_sampling
        self.dapo_reward = dapo_reward
        self._dapo_cached_rollout = None
        super().__init__(*args, **kwargs)
        if self.args.loss_type != 'dapo' or self.args.beta != 0:
            raise ValueError('DAPO requires token-level dapo loss and beta=0')
        if self.use_vllm or self.tools or self.environment_factories or self.rollout_func:
            raise ValueError('This DAPO profile supports native text-only generation')
        if self.args.scale_rewards != 'group' or self.args.epsilon_high <= self.args.epsilon:
            raise ValueError('DAPO requires group advantages and asymmetric Clip-Higher')
        if self.args.steps_per_generation != self.args.gradient_accumulation_steps:
            raise ValueError('Use one generation batch per accumulated optimizer update')
        self._sampling_rows = self.train_dataset.to_list()
        self.add_callback(DAPOProgressCallback())
        self.sampling_path = Path(self.args.output_dir) / 'dynamic_sampling.jsonl'
        self.candidates_path = Path(self.args.output_dir) / 'sampling_groups.jsonl'
        if self.is_world_process_zero():
            self.sampling_path.write_text('', encoding='utf-8')
            self.candidates_path.write_text('', encoding='utf-8')

    def _generate(self, prompts):
        if self._dapo_cached_rollout is not None:
            expected, rollout = self._dapo_cached_rollout
            self._dapo_cached_rollout = None
            if prompts != expected:
                raise RuntimeError('Cached DAPO rollouts lost prompt alignment')
            return rollout
        return super()._generate(prompts)

    def _replacement_inputs(self, local_count, seen_ids, rng):
        workers = self.accelerator.num_processes
        target = local_count * workers // self.num_generations
        # Every rank uses the same RNG and train-only dataset; no dev/test access.
        available = [i for i, row in enumerate(self._sampling_rows) if row['id'] not in seen_ids]
        if len(available) < target:
            raise RuntimeError('DAPO exhausted distinct train prompts while filling a batch')
        indices = rng.sample(available, target)
        rows = [dict(self._sampling_rows[i]) for i in indices]
        global_inputs = [row for row in rows for _ in range(self.num_generations)]
        rank = self.accelerator.process_index
        return global_inputs[rank * local_count:(rank + 1) * local_count]

    def _generate_and_score_completions(self, inputs):
        if not self.model.training:
            return super()._generate_and_score_completions(inputs)
        local_count = len(inputs)
        target = local_count * self.accelerator.num_processes
        if target != self.args.generation_batch_size or target % self.num_generations:
            raise ValueError('DAPO generation batch disagrees with TRL configuration')
        selected, seen_ids = [], set()
        candidate_groups = accepted_groups = 0
        rng = random.Random(self.args.seed + 1000003 * self.state.global_step)
        max_batches = self.dynamic_sampling['max_generation_batches']
        for attempt in range(1, max_batches + 1):
            if attempt > 1:
                inputs = self._replacement_inputs(local_count, seen_ids, rng)
            prompts = [r['prompt'] for r in inputs]
            raw = super()._generate(prompts)
            prompt_ids, completion_ids, tool_mask, completions, logps, sampling_mask, extra, images, tool_images = raw
            if any(v is not None for v in (tool_mask, logps, sampling_mask, images)) or extra or any(tool_images):
                raise RuntimeError('Unexpected non-text rollout fields in DAPO')
            records = []
            eos = {*self.eos_token_ids, self._tokenizer.pad_token_id}
            for example, pid, cid, completion in zip(inputs, prompt_ids, completion_ids, completions, strict=True):
                records.append({'example': example, 'prompt_ids': pid, 'completion_ids': cid,
                                'completion': completion, 'terminated': bool(cid) and cid[-1] in eos,
                                'score': self.dapo_reward.assess(completion, example['reference'], len(cid))})
            records = gather_records(records)
            seen_ids.update(r['example']['id'] for r in records)
            accepted, diagnostics = filter_groups(records, self.num_generations)
            selected.extend(accepted)
            candidate_groups += len(diagnostics)
            accepted_groups += len(accepted) // self.num_generations
            if self.is_world_process_zero():
                with self.candidates_path.open('a', encoding='utf-8') as handle:
                    for row in diagnostics:
                        handle.write(json.dumps({'step': self.state.global_step, 'attempt': attempt, **row}) + '\n')
                print(f'DAPO sampling: step={self.state.global_step} attempt={attempt} '
                      f'usable_groups={len(selected) // self.num_generations}/{target // self.num_generations}', flush=True)
            if len(selected) >= target:
                break
        else:
            raise RuntimeError(f'DAPO could not fill a nonzero-signal batch after {max_batches} generation batches')
        # Drop surplus WHOLE groups. Rewards and advantages are recalculated on
        # the selected batch by native TRL, whose dapo loss counts accepted tokens.
        selected = selected[:target]
        summary = {'step': self.state.global_step, 'generation_batches': attempt,
                   'candidate_groups': candidate_groups, 'qualified_groups': accepted_groups,
                   'selected_groups': target // self.num_generations,
                   'acceptance_rate': accepted_groups / candidate_groups,
                   'selected_ids': [r['example']['id'] for r in selected[::self.num_generations]]}
        if self.is_world_process_zero():
            with self.sampling_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(summary) + '\n')
        for key in ('generation_batches', 'acceptance_rate', 'candidate_groups', 'selected_groups'):
            self._metrics['train'][f'dapo/{key}'].append(summary[key])
        rank = self.accelerator.process_index
        selected = selected[rank * local_count:(rank + 1) * local_count]
        selected_inputs = [r['example'] for r in selected]
        selected_prompts = [r['prompt'] for r in selected_inputs]
        rollout = ([r['prompt_ids'] for r in selected], [r['completion_ids'] for r in selected],
                   None, [r['completion'] for r in selected], None, None, {}, None, [])
        self._dapo_cached_rollout = (selected_prompts, rollout)
        try:
            return super()._generate_and_score_completions(selected_inputs)
        finally:
            self._dapo_cached_rollout = None
