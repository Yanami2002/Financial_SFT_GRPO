"""Generate committed smoke and equal-data continued-SFT control configurations."""
from pathlib import Path
import yaml

def main():
    root = Path(__file__).resolve().parents[1]
    load = lambda name: yaml.safe_load((root / 'configs' / (name + '.yaml')).read_text())
    def save(name, cfg):
        (root / 'configs' / (name + '.yaml')).write_text(yaml.safe_dump(cfg, sort_keys=False))
    s = load('sft')
    s.update(limit=16)
    s['training'].update(output_dir='outputs/qwen3.5-2b/smoke-sft', max_steps=1, warmup_steps=0.0, gradient_accumulation_steps=1, save_steps=1, eval_steps=1)
    s['validation']['limit'] = 4
    s['validation']['generation']['max_new_tokens'] = 48
    save('sft_smoke', s)
    g = load('grpo')
    g.update(limit=16, sft_adapter='outputs/qwen3.5-2b/smoke-sft')
    g['training'].update(output_dir='outputs/qwen3.5-2b/smoke-grpo', max_steps=4, warmup_steps=0.0, gradient_accumulation_steps=4,
                         num_generations=4, max_completion_length=48, save_steps=4, eval_steps=4)
    g['validation']['limit'] = 4
    g['validation']['generation']['max_new_tokens'] = 48
    save('grpo_smoke', g)
    e = load('eval')
    e.update(data='data/processed/dev.jsonl', split='dev', limit=4, output_dir='results/qwen3.5-2b/smoke-dev')
    e['variants']['sft']['adapter'] = 'outputs/qwen3.5-2b/smoke-sft'
    e['variants']['grpo']['adapter'] = 'outputs/qwen3.5-2b/smoke-grpo'
    e['generation']['max_new_tokens'] = 48
    save('eval_smoke', e)
    # train_sft and train_grpo have exactly the same filtered train IDs.
    control = load('sft')
    control.update(init_adapter='outputs/qwen3.5-2b/sft', data='data/processed/qwen35_control_sft.jsonl')
    control['training'].update(output_dir='outputs/qwen3.5-2b/continued-sft', max_steps=-1, num_train_epochs=1.0, learning_rate=0.000005)
    save('continued_sft', control)
    equal = load('grpo')
    equal['data'] = 'data/processed/qwen35_control_grpo.jsonl'
    equal['training']['output_dir'] = 'outputs/qwen3.5-2b/grpo-equal-data'
    save('grpo_equal_data', equal)

if __name__ == '__main__':
    main()
