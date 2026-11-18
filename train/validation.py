"""Greedy dev accuracy with native Transformers best-checkpoint retention.

References: https://huggingface.co/docs/transformers/main_classes/trainer
https://github.com/huggingface/trl/blob/main/trl/scripts/grpo.py
Only imported when constructing a real trainer; CLI/dry-run stays CPU/minimal.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import Trainer
from trl import GRPOTrainer

from eval.metrics import evaluate_prediction, aggregate
from financeqa.common import digest, record_run, write_json, write_jsonl
from financeqa.distributed import gather_records
from financeqa.modeling import generate_answer, adapter_fingerprint


@contextmanager
def evaluation_state(model, tokenizer, seed):
    """Restore mode, cache, padding, and RNG even if generation raises."""
    was_training, cache = model.training, model.config.use_cache
    padding = tokenizer.padding_side
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [model.device.index] if model.device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            # Seed CPU and only this worker's assigned CUDA device.
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(model.device):
                    torch.cuda.manual_seed(seed)
            model.eval()
            model.config.use_cache = True
            tokenizer.padding_side = 'left'
            yield
    finally:
        model.train(was_training)
        model.config.use_cache = cache
        tokenizer.padding_side = padding
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


class NumericalEvaluationMixin:
    def __init__(self, *args, validation_config, model_spec, **kwargs):
        self.validation_config = validation_config
        self.model_spec = model_spec
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        """Score generated answers, never teacher-forced tokens or GRPO rewards."""
        rows = self.eval_dataset if eval_dataset is None else eval_dataset
        if not rows or any(r.get('split') != 'dev' for r in rows):
            raise ValueError('Checkpoint selection only evaluates dev records')
        start = time.perf_counter()
        self._memory_tracker.start()
        model = self.accelerator.unwrap_model(self.model_wrapped)
        tokenizer = self.processing_class
        rank, workers = self.accelerator.process_index, self.accelerator.num_processes
        local = []
        with evaluation_state(model, tokenizer, self.args.seed):
            for index in range(rank, len(rows), workers):
                row = rows[index]
                result = generate_answer(model, tokenizer, self.model_spec, row['prompt'],
                                         self.validation_config['generation'])
                record = evaluate_prediction(row, result['output'], **self.validation_config['tolerance'])
                record.update({k:v for k,v in result.items() if k != 'output'})
                local.append({'index':index, 'record':record})
        # Unequal shard lengths are allowed: unwrap DDP for generation, then gather once.
        gathered = sorted(gather_records(local), key=lambda item:item['index'])
        if [item['index'] for item in gathered] != list(range(len(rows))):
            raise RuntimeError('Distributed dev evaluation lost or duplicated records')
        predictions = [item['record'] for item in gathered]
        summary = aggregate(predictions)
        if summary['numerical_accuracy'] is None:
            raise ValueError('No scorable dev records for checkpoint selection')
        elapsed = time.perf_counter()-start
        metrics = {f'{metric_key_prefix}_{k}':summary[k] for k in
                   ('numerical_accuracy', 'parseable_rate', 'n', 'n_scorable', 'n_unscorable', 'correct')}
        metrics[f'{metric_key_prefix}_runtime'] = elapsed
        if self.is_world_process_zero():
            step = self.state.global_step
            directory = Path(self.args.output_dir)/'dev-evaluations'/f'step-{step}'
            checkpoint = Path(self.args.output_dir)/f'checkpoint-{step}'
            cfg = {'model':self.model_spec, 'seed':self.args.seed, 'split':'dev',
                   **self.validation_config, 'output_dir':str(directory.parent),
                   'variants':{f'step{step}':{'adapter':str(checkpoint)}}}
            data = Path(self.validation_config['data'])
            files = [data]
            manifest = data.parent/'manifest.json'
            if manifest.exists():
                files.append(manifest)
            provenance = record_run(directory, cfg, files)
            provenance.update(variant=f'step{step}', checkpoint_pending_save=True)
            write_json(directory/'run.json', provenance)
            summary['protocol'] = {'data_sha256':digest(data), 'ids':[r['id'] for r in rows],
                'prompt_sha256':provenance['prompt_sha256'], 'split':'dev',
                'generation':cfg['generation'], 'tolerance':cfg['tolerance'],
                'model':self.model_spec, 'seed':self.args.seed}
            write_json(directory/'metrics.json', summary)
            write_jsonl(directory/'predictions.jsonl', predictions)
            history = Path(self.args.output_dir)/'dev_history.jsonl'
            with history.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'step':step, **metrics}, allow_nan=False)+'\n')
            print(f'Dev check at step {step}: {summary["correct"]}/{summary["n_scorable"]} correct '
                  f'({summary["numerical_accuracy"]:.2%})', flush=True)
        # Bypass GRPO.log: benchmark evaluation must not drain its train reward buffers.
        Trainer.log(self, metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    def _save_checkpoint(self, model, trial):
        # Native Trainer updates best_model_checkpoint BEFORE rotating old checkpoints.
        super()._save_checkpoint(model, trial)
        if self.is_world_process_zero():
            directory = Path(self.args.output_dir)/'dev-evaluations'/f'step-{self.state.global_step}'
            run = directory/'run.json'
            if run.exists():
                provenance = json.loads(run.read_text())
                checkpoint = Path(self.args.output_dir)/f'checkpoint-{self.state.global_step}'
                provenance.update(adapter=adapter_fingerprint(checkpoint), checkpoint_pending_save=False)
                write_json(run, provenance)
            self.write_selection(exported=False)

    def write_selection(self, exported=True):
        """Explain which saved version was selected and what the final export contains."""
        if not self.is_world_process_zero():
            return
        checkpoint = self.state.best_model_checkpoint
        if not checkpoint:
            raise RuntimeError('Training completed without a scored dev checkpoint')
        step = self.state.best_global_step
        write_json(Path(self.args.output_dir)/'best_checkpoint.json', {
            'selection_split':'dev', 'metric':'numerical_accuracy', 'greater_is_better':True,
            'tie_break':'earliest evaluated step',
            'selected':{'adapter':checkpoint, 'step':step, 'accuracy':self.state.best_metric,
                        'metrics':str(Path(self.args.output_dir)/'dev-evaluations'/f'step-{step}'/'metrics.json')},
            'exported_adapter':str(self.args.output_dir) if exported else None,
            'export_contains_best':exported,
        })


class NumericalSFTTrainer(NumericalEvaluationMixin, Trainer):
    pass


class NumericalGRPOTrainer(NumericalEvaluationMixin, GRPOTrainer):
    pass
