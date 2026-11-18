"""Dev-only selection, true generated accuracy, and real best-weight retention."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap
import time
from types import MethodType

import pytest
import yaml

from financeqa.common import validation_data, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def dev_row(key='dev-a'):
    return {'id':key, 'split':'dev', 'question':'question',
            'prompt':[{'role':'user', 'content':'question'}],
            'reference':{'value':.2, 'unit':'ratio'}}


def config_for(tmp_path, rows):
    cfg = yaml.safe_load((ROOT/'configs/sft.yaml').read_text())
    path = tmp_path/'dev.jsonl'
    write_jsonl(path, rows)
    cfg['validation']['data'] = str(path)
    return cfg


@pytest.mark.parametrize('split', ['train', 'test'])
def test_selection_rejects_non_dev_even_outside_limit(tmp_path, split):
    cfg = config_for(tmp_path, [dev_row(), {**dev_row('other'), 'split':split}])
    cfg['validation']['limit'] = 1
    with pytest.raises(ValueError, match='never train/test'):
        validation_data(cfg, [])


def test_selection_rejects_duplicates_overlap_unscorable_and_answers(tmp_path):
    cfg = config_for(tmp_path, [dev_row(), dev_row()])
    with pytest.raises(ValueError, match='Duplicate'):
        validation_data(cfg, [])
    cfg = config_for(tmp_path, [dev_row()])
    with pytest.raises(ValueError, match='overlap'):
        validation_data(cfg, [{'id':'dev-a'}])
    cfg = config_for(tmp_path, [{**dev_row(), 'reference':None}])
    with pytest.raises(ValueError, match='No scorable'):
        validation_data(cfg, [])
    row = dev_row()
    row['prompt'].append({'role':'assistant','content':'reference answer'})
    with pytest.raises(ValueError, match='without reference answers'):
        validation_data(config_for(tmp_path, [row]), [])


@pytest.mark.parametrize('change', [
    {'eval_steps':50}, {'load_best_model_at_end':False},
    {'metric_for_best_model':'loss'}, {'greater_is_better':False}, {'save_total_limit':1},
])
def test_selection_rejects_unsafe_checkpoint_settings(tmp_path, change):
    cfg = config_for(tmp_path, [dev_row()])
    cfg['training'].update(change)
    with pytest.raises(ValueError):
        validation_data(cfg, [])


def test_evaluation_state_restores_rng_and_mode_after_failure():
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')
    pytest.importorskip('trl')
    from types import SimpleNamespace
    from train.validation import evaluation_state
    model = torch.nn.Linear(1,1)
    model.config = SimpleNamespace(use_cache=False)
    model.device = torch.device('cpu')
    tok = SimpleNamespace(padding_side='right')
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with pytest.raises(RuntimeError, match='fixture failure'):
        with evaluation_state(model, tok, 123):
            assert not model.training and model.config.use_cache and tok.padding_side == 'left'
            random.random(); np.random.rand(); torch.rand(1)
            raise RuntimeError('fixture failure')
    assert model.training and model.config.use_cache is False and tok.padding_side == 'right'
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)


def tiny_model_and_tokenizer():
    pytest.importorskip('torch')
    pytest.importorskip('trl')
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from peft import get_peft_model, LoraConfig
    raw = Tokenizer(WordLevel({'[PAD]':0, '[EOS]':1, '[UNK]':2,
        '{"answer":20,"unit":"percent"}':3, '{"answer":0,"unit":"percent"}':4,
        'question':5}, unk_token='[UNK]'))
    raw.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    tok.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    cfg = Qwen3_5TextConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        layer_types=['full_attention'], max_position_embeddings=128, eos_token_id=1, pad_token_id=0)
    model = get_peft_model(Qwen3_5ForCausalLM(cfg),
        LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4, target_modules=['q_proj','v_proj']))
    model.config.use_cache = False
    return model, tok


@pytest.mark.parametrize('steps,interval,correct_counts,best_step', [
    (4,1,[1,2,0,1],2),  # Earlier best survives rotation; the last version is worse.
    (3,2,[1,2],3),      # The final step must be checked even outside the usual interval.
    (4,1,[2,2,0,0],1),  # Ties retain the first version, including during checkpoint rotation.
])
def test_native_best_retention_and_export(tmp_path, steps, interval, correct_counts, best_step):
    torch = pytest.importorskip('torch')
    pytest.importorskip('trl')
    from datasets import Dataset
    from transformers import TrainingArguments
    from safetensors.torch import load_file
    from train.sft import AssistantCollator
    from train.validation import NumericalSFTTrainer
    from financeqa.distributed import save_training
    model, tok = tiny_model_and_tokenizer()
    cfg = config_for(tmp_path, [dev_row(), dev_row('dev-b')])
    cfg['validation']['generation'].update(max_input_tokens=16, max_new_tokens=8)
    calls = []

    def generate(self, input_ids, **kwargs):
        # Explicit test-only outputs; training forward/backward and saving are real.
        check, item = divmod(len(calls), 2)
        calls.append(input_ids.tolist())
        token = 3 if item < correct_counts[check] else 4
        return torch.cat([input_ids, torch.tensor([[token]], device=input_ids.device)], dim=1)

    model.generate = MethodType(generate, model)
    out = tmp_path/'training'
    args = TrainingArguments(output_dir=str(out), use_cpu=True, bf16=False, max_steps=steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=1, learning_rate=.01,
        eval_strategy='steps', eval_steps=interval, save_strategy='steps', save_steps=interval,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='numerical_accuracy',
        greater_is_better=True, report_to='none', disable_tqdm=True, logging_steps=1)
    ds = Dataset.from_list([{'input_ids':[5,5,1], 'attention_mask':[1,1,1], 'labels':[-100,5,1]}]*2)
    trainer = NumericalSFTTrainer(model=model, args=args, train_dataset=ds,
        eval_dataset=[dev_row(), dev_row('dev-b')], processing_class=tok,
        data_collator=AssistantCollator(tok.pad_token_id), validation_config=cfg['validation'],
        model_spec={'name':'tiny-random-test-fixture', 'revision':'fixture', 'chat_template_kwargs':{'enable_thinking':False}})
    start = time.perf_counter()
    result = trainer.train()
    save_training(trainer, tok, out, result, start)
    trainer.write_selection()
    selection = json.loads((out/'best_checkpoint.json').read_text())
    assert selection['selected']['step'] == best_step
    assert selection['selected']['accuracy'] == 1
    assert selection['selection_split'] == 'dev' and selection['export_contains_best']
    assert len(list(out.glob('checkpoint-*'))) == 2
    assert (out/f'checkpoint-{best_step}').is_dir() and (out/f'checkpoint-{steps}').is_dir()
    best = load_file(str(out/f'checkpoint-{best_step}'/'adapter_model.safetensors'))
    final = load_file(str(out/'adapter_model.safetensors'))
    assert best.keys() == final.keys() and all(torch.equal(best[k],final[k]) for k in best)
    assert len(calls) == 2*len(correct_counts)
    for directory in (out/'dev-evaluations').iterdir():
        run = json.loads((directory/'run.json').read_text())
        assert run['checkpoint_pending_save'] is False
        assert run['adapter']


def test_overlong_dev_is_counted_and_grpo_reward_buffers_are_untouched(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('trl')
    from trl import GRPOConfig
    from train.validation import NumericalGRPOTrainer
    from rewards.numerical import numerical_reward
    from datasets import Dataset
    model, tok = tiny_model_and_tokenizer()
    cfg = config_for(tmp_path, [dev_row(), {**dev_row('long'),
        'prompt':[{'role':'user','content':'question '*40}]}])
    cfg['validation']['generation'].update(max_input_tokens=16, max_new_tokens=8)
    train = Dataset.from_list([{**dev_row('train'), 'split':'train'}])
    args = GRPOConfig(output_dir=str(tmp_path/'grpo'), use_cpu=True, bf16=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=2, num_generations=2,
        num_generations_eval=1, eval_strategy='steps', eval_steps=1, save_steps=1,
        load_best_model_at_end=True, metric_for_best_model='numerical_accuracy', greater_is_better=True,
        beta=.04, gradient_checkpointing=False, report_to='none', disable_tqdm=True)
    trainer = NumericalGRPOTrainer(model=model, args=args, processing_class=tok, train_dataset=train,
        eval_dataset=validation_data(cfg, []), reward_funcs=numerical_reward,
        validation_config=cfg['validation'], model_spec={'chat_template_kwargs':{'enable_thinking':False}})

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, torch.tensor([[3]], device=input_ids.device)], dim=1)

    model.generate = MethodType(generate, model)
    trainer._metrics['train']['fixture'] = [7.]
    refs = {n:p.detach().clone() for n,p in model.named_parameters() if '.ref.' in n}
    assert refs
    model.train()
    metrics = trainer.evaluate()
    assert metrics['eval_n_scorable'] == 2 and metrics['eval_numerical_accuracy'] == .5
    assert trainer._metrics['train']['fixture'] == [7.]
    assert model.training and model.config.use_cache is False
    assert all(torch.equal(model.get_parameter(n),v) for n,v in refs.items())
    predictions = [json.loads(line) for line in (tmp_path/'grpo/dev-evaluations/step-0/predictions.jsonl').read_text().splitlines()]
    assert predictions[1]['error'] == 'context_too_long'
    assert predictions[1]['score']['scorable'] and not predictions[1]['score']['correct']


def test_three_rank_dev_shards_are_gathered_once(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('trl')
    worker = tmp_path/'worker.py'
    worker.write_text(textwrap.dedent('''
        import json, os, sys
        from pathlib import Path
        from types import MethodType
        import torch
        from datasets import Dataset
        from transformers import TrainingArguments
        sys.path.insert(0, str(Path.cwd()/'tests'))
        from test_validation import tiny_model_and_tokenizer, dev_row
        from train.validation import NumericalSFTTrainer
        model, tok = tiny_model_and_tokenizer()
        def generate(self, input_ids, **kwargs):
            return torch.cat([input_ids, torch.tensor([[3]], device=input_ids.device)], dim=1)
        model.generate = MethodType(generate, model)
        rows = [dev_row('a'), dev_row('b'),
                {**dev_row('long'), 'prompt':[{'role':'user','content':'question '*40}]},
                {**dev_row('unscorable'), 'reference':None}]
        out = Path(sys.argv[1])
        args = TrainingArguments(output_dir=str(out), use_cpu=True, bf16=False,
            per_device_train_batch_size=1, report_to='none', disable_tqdm=True,
            eval_strategy='steps', eval_steps=1, save_steps=1,
            load_best_model_at_end=True, metric_for_best_model='numerical_accuracy', greater_is_better=True)
        trainer = NumericalSFTTrainer(model=model, args=args, processing_class=tok,
            train_dataset=Dataset.from_list([{'input_ids':[5,1], 'labels':[-100,1]}]),
            eval_dataset=rows, validation_config={'data':str(out/'dev.jsonl'), 'limit':None,
                'generation':{'max_input_tokens':16,'max_new_tokens':8,'do_sample':False},
                'tolerance':{'atol':1e-6,'rtol':.001}},
            model_spec={'chat_template_kwargs':{'enable_thinking':False}})
        metrics = trainer.evaluate()
        (out/f'rank-{os.environ["RANK"]}.json').write_text(json.dumps(metrics))
        trainer.accelerator.wait_for_everyone()
        trainer.end()
    '''))
    rows = [dev_row('a'), dev_row('b'),
            {**dev_row('long'), 'prompt':[{'role':'user','content':'question '*40}]},
            {**dev_row('unscorable'), 'reference':None}]
    write_jsonl(tmp_path/'dev.jsonl', rows)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
           'TOKENIZERS_PARALLELISM':'false', 'PYTHONPATH':str(ROOT)}
    result = subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',
        '--nnodes=1','--nproc_per_node=3',str(worker),str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    for rank in range(3):
        metrics = json.loads((tmp_path/f'rank-{rank}.json').read_text())
        assert metrics['eval_n'] == 4 and metrics['eval_n_scorable'] == 3
        assert metrics['eval_numerical_accuracy'] == 2/3
    records = [json.loads(line) for line in (tmp_path/'dev-evaluations/step-0/predictions.jsonl').read_text().splitlines()]
    assert [r['id'] for r in records] == ['a','b','long','unscorable']
