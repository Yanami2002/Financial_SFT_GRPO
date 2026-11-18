"""Validate planned DDP configs and optionally the real tokenizer; never load model weights."""
import argparse
from pathlib import Path
from financeqa.common import training_rows, validation_data, write_json, digest
from financeqa.distributed import validate_batch
from financeqa.modeling import chat_kwargs, load_tokenizer, render_chat
from train.sft import tokenize_example


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--world-size', type=int, default=3, help='Planned training workers; no workers are launched')
    p.add_argument('--tokenizer', action='store_true', help='Download/read config and tokenizer metadata only')
    p.add_argument('--output', default='results/qwen3.5-2b/setup.json')
    a = p.parse_args()
    import yaml
    configs = {name:yaml.safe_load(Path(f'configs/{name}.yaml').read_text()) for name in ('sft','grpo','eval')}
    assert configs['sft']['model'] == configs['grpo']['model'] == configs['eval']['model'], 'Model settings differ across stages'
    report = {'model':configs['sft']['model'], 'loads_model_weights':False, 'runs_training':False,
              'batch':{name:validate_batch(configs[name]['training'],grpo=(name=='grpo'),workers=a.world_size) for name in ('sft','grpo')}}
    for name in ('sft','grpo'):
        rows = training_rows(configs[name]['data'])
        dev = validation_data(configs[name], rows)
        report[name] = {'samples':len(rows), 'data_sha256':digest(configs[name]['data']),
                        'dev_samples':len(dev), 'dev_sha256':digest(configs[name]['validation']['data']),
                        'eval_every_steps':configs[name]['training']['eval_steps'],
                        'selection_metric':'numerical_accuracy'}
    if a.tokenizer:
        from transformers import AutoConfig, Qwen3_5ForCausalLM
        from trl import GRPOConfig
        from transformers import TrainingArguments
        from dataclasses import fields
        spec = configs['sft']['model']
        for name, cls in [('sft',TrainingArguments),('grpo',GRPOConfig)]:
            unknown = set(configs[name]['training']) - {f.name for f in fields(cls)}
            if unknown:
                raise ValueError(f'{name}: unsupported arguments {unknown}')
        config = AutoConfig.from_pretrained(spec['name'],revision=spec['revision'],cache_dir=spec['cache_dir'])
        tok = load_tokenizer(spec)
        row = training_rows(configs['sft']['data'],1)[0]
        enc = tokenize_example(row,tok,configs['sft']['max_length'],chat_kwargs(spec))
        if enc is None:
            raise ValueError('First example exceeds token budget; use a fitting example to check setup')
        supervised = tok.decode([v for v in enc['labels'] if v!=-100])
        if '<think>' in supervised or '</think>' in supervised:
            raise ValueError('Empty thinking prefix must be masked, not supervised')
        report['tokenizer'] = {'model_type':config.model_type, 'text_model_type':config.get_text_config().model_type,
             'loader_class':Qwen3_5ForCausalLM.__name__, 'supervised_text':supervised,
             'prompt_tokens':len(render_chat(tok,row['messages'][:-1],generation=True,template_kwargs=chat_kwargs(spec))),
             'eos_supervised':tok.eos_token_id in enc['labels'], 'assistant_mask_verified':True}
    write_json(a.output,report)
    print(report)

if __name__=='__main__':
    main()
