import json
from pathlib import Path
import subprocess
import sys
import pytest
from train.sft import tokenize_example, AssistantCollator
from eval.metrics import evaluate_prediction, aggregate
from train.grpo import DiagnosticReward

ROOT = Path(__file__).resolve().parents[1]


class TemplateTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        # Deterministic stand-in for template boundary testing, no network.
        tokens = []
        for msg in messages:
            tokens += [1] + list(msg['content'].encode()) + [2]
        if add_generation_prompt:
            tokens += [1]
        return tokens


def test_assistant_mask_eos_padding_and_no_truncation():
    pytest.importorskip('torch')
    row = {'messages':[{'role':'system','content':'S'}, {'role':'user','content':'Q'},
                       {'role':'assistant','content':'ABC'}]}
    tok = TemplateTokenizer()
    enc = tokenize_example(row, tok, 100)
    n = len(tok.apply_chat_template(row['messages'][:-1], True, True))
    assert enc['labels'][:n] == [-100]*n
    assert enc['labels'][n:] == enc['input_ids'][n:] and enc['labels'][-1] == 2
    assert tokenize_example(row, tok, 2) is None
    short = {k:v[:-1] for k,v in enc.items()}
    batch = AssistantCollator(2)([enc, short])
    assert batch['labels'][1,-1] == -100 and batch['attention_mask'][1,-1] == 0
    assert batch['labels'][0,-1] == 2  # EOS gets loss even when PAD==EOS


def test_metrics_denominators_and_group_diagnostic(tmp_path):
    row = {'id':'a','split':'dev','question':'q','reference':{'value':.2,'unit':'ratio'}}
    a = evaluate_prediction(row, '{"answer":20,"unit":"percent"}')
    b = evaluate_prediction({**row, 'id':'b'}, 'bad')
    c = evaluate_prediction({**row, 'id':'c', 'reference':None}, '{}')
    metrics = aggregate([a,b,c])
    assert metrics['numerical_accuracy'] == .5
    assert metrics['n_unscorable'] == 1 and metrics['parseable_rate'] == 1/3
    reward = DiagnosticReward(tmp_path/'groups.jsonl', {'atol':1e-6, 'rtol':.001})
    assert reward(['{}','{}'], [row['reference']]*2, ['a','a']) == [0,0]
    info = json.loads((tmp_path/'groups.jsonl').read_text())
    assert info['std'] == 0 and info['n'] == 2


@pytest.mark.parametrize('module', ['data.prepare_data','train.sft','train.grpo','eval.evaluate',
                                  'eval.compare','inference.predict','demo.app'])
def test_cli_help(module):
    result = subprocess.run([sys.executable,'-m',module,'--help'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_cli_cpu_dry_run(tmp_path):
    p = tmp_path/'input.jsonl'
    p.write_text(json.dumps({'context':'Revenue: 100 to 120 USD','question':'Percentage growth?'})+'\n')
    out = subprocess.run([sys.executable,'-m','inference.predict','--input',str(p),
                          '--output',str(tmp_path/'pred.jsonl'),'--dry-run'], cwd=ROOT,capture_output=True,text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)['n'] == 1


def test_checkpoint_selection_rejects_test(tmp_path):
    from eval.select_checkpoint import select
    p = tmp_path/'metrics.json'
    p.write_text(json.dumps({'protocol': {'split':'test'}, 'numerical_accuracy':1.}))
    with pytest.raises(ValueError, match='never test'):
        select([p])


def test_compare_rejects_mismatched_protocols(tmp_path):
    a, b = tmp_path/'a.json', tmp_path/'b.json'
    a.write_text(json.dumps({'protocol':{'seed':42}}))
    b.write_text(json.dumps({'protocol':{'seed':43}}))
    result = subprocess.run([sys.executable,'-m','eval.compare',str(a),str(b)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode != 0 and 'Comparison rejected' in result.stderr


def test_frozen_reference_precision_alignment():
    torch = pytest.importorskip('torch')
    from train.grpo import align_reference_adapter
    model = torch.nn.Module()
    model.adapters = torch.nn.ModuleDict({'default':torch.nn.Linear(2,2),
                                          'ref':torch.nn.Linear(2,2,dtype=torch.bfloat16)})
    for p, r in zip(model.adapters['default'].parameters(),model.adapters['ref'].parameters()):
        r.data.copy_(p.data); r.requires_grad_(False)
    result = align_reference_adapter(model)
    assert result['initial_reference_equals_sft'] and result['precision_aligned_tensor_count'] == 2
    for p, r in zip(model.adapters['default'].parameters(),model.adapters['ref'].parameters()):
        assert torch.equal(p,r) and p.dtype == r.dtype and not r.requires_grad
