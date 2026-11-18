"""Write identical train IDs for continued SFT and GRPO under BOTH token budgets."""
import argparse
from pathlib import Path
from financeqa.common import training_rows, write_jsonl, write_json
from train.sft import tokenize_example
from financeqa.modeling import load_tokenizer, render_chat, chat_kwargs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sft-config',default='configs/continued_sft.yaml')
    p.add_argument('--grpo-config',default='configs/grpo.yaml')
    a = p.parse_args()
    import yaml
    s, g = [yaml.safe_load(Path(v).read_text()) for v in (a.sft_config,a.grpo_config)]
    if s['model'] != g['model']:
        raise ValueError('Control cohort requires identical model and chat settings')
    tok = load_tokenizer(g['model'])
    template_kwargs = chat_kwargs(g['model'])
    srows = training_rows('data/processed/train_sft.jsonl')
    grows = {r['id']:r for r in training_rows('data/processed/train_grpo.jsonl')}
    selected = [r for r in srows if r['id'] in grows
                and tokenize_example(r,tok,s['max_length'],template_kwargs) is not None
                and len(render_chat(tok,grows[r['id']]['prompt'],generation=True,template_kwargs=template_kwargs)) <= g['max_prompt_tokens']]
    if not selected:
        raise ValueError('No common train examples fit both budgets')
    write_jsonl('data/processed/qwen35_control_sft.jsonl',selected)
    write_jsonl('data/processed/qwen35_control_grpo.jsonl',[grows[r['id']] for r in selected])
    write_json('data/processed/qwen35_control_manifest.json',{'ids':[r['id'] for r in selected],
                'model':g['model'], 'sft_max_length':s['max_length'],'grpo_max_prompt_tokens':g['max_prompt_tokens'],
                'note':'Same prompts/IDs; sample exposures and compute require separate matching.'})
    print({'shared_train_ids':len(selected)})

if __name__ == '__main__':
    main()
