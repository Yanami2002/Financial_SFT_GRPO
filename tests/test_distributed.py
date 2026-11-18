import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock
import pytest
from financeqa.distributed import validate_batch
from train.grpo import diagnostic_groups, DiagnosticReward

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('workers',[1,2,3])
def test_batch_divisibility(workers):
    cfg={'per_device_train_batch_size':1,'gradient_accumulation_steps':8,'num_generations':8}
    result=validate_batch(cfg,grpo=True,workers=workers)
    assert result['effective_batch_size']==workers*8
    assert result['prompts_per_generation_batch']==workers


def test_bad_generation_batch():
    with pytest.raises(ValueError,match='divisible'):
        validate_batch({'per_device_train_batch_size':1,'gradient_accumulation_steps':2,'num_generations':4},grpo=True,workers=3)


def test_diagnostics_preserve_repeated_id_groups():
    records=[{'id':'same','score':{'correct':False,'parseable':True}}]*8
    groups=diagnostic_groups(records,4)
    assert len(groups)==2 and all(len(v)==4 for _,v in groups)
    with pytest.raises(ValueError,match='order'):
        diagnostic_groups([{'id':str(i),'score':{}} for i in range(4)],4)


def test_local_rank_device_selection(monkeypatch):
    torch=pytest.importorskip('torch')
    from financeqa.modeling import resolve_device
    monkeypatch.setenv('WORLD_SIZE','3'); monkeypatch.setenv('LOCAL_RANK','2')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    selected=[]
    monkeypatch.setattr(torch.cuda,'set_device',selected.append)
    assert resolve_device({'device':'auto'},trainable=True)=='cuda:2'
    assert selected==['cuda:2']
    monkeypatch.delenv('LOCAL_RANK')
    with pytest.raises(RuntimeError,match='LOCAL_RANK'):
        resolve_device({'device':'auto'},trainable=True)


def test_reward_nonzero_rank_does_not_touch_log(tmp_path,monkeypatch):
    monkeypatch.setattr('train.grpo.is_main_process',lambda:False)
    monkeypatch.setattr('train.grpo.gather_records',lambda records:records)
    p=tmp_path/'groups.jsonl'
    reward=DiagnosticReward(p,{'atol':1e-6,'rtol':.001},num_generations=2)
    assert reward(['{}','{}'],[{'value':.2,'unit':'ratio'}]*2,['a','a'])==[0,0]
    assert not p.exists()


def test_final_save_only_main_process(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    from financeqa.distributed import save_training
    model=Mock(); tokenizer=Mock();trainer=Mock()
    trainer.is_world_process_zero.return_value=False
    monkeypatch.setattr('financeqa.modeling.resource_finish',lambda start:{'seconds':1})
    monkeypatch.setattr('financeqa.distributed.gather_records',lambda rows:rows)
    save_training(trainer,tokenizer,tmp_path,Mock(metrics={}),0)
    tokenizer.save_pretrained.assert_not_called()
    assert not (tmp_path/'train_metrics.json').exists()
    trainer.is_world_process_zero.return_value=True
    trainer.accelerator.unwrap_model.return_value=model
    save_training(trainer,tokenizer,tmp_path,Mock(metrics={}),0)
    model.save_pretrained.assert_called_once_with(tmp_path,selected_adapters=['default'])
    assert json.loads((tmp_path/'train_metrics.json').read_text())['resources_by_rank'][0]['seconds']==1


def test_three_process_cpu_group_probe(tmp_path):
    pytest.importorskip('torch')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc_per_node=3','-m','scripts.check_distributed','--output',str(tmp_path)],
        cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['cpu_only'] and not report['runs_training']
    assert report['group_sizes']==[6,6] and report['single_writer_verified']


def test_all_committed_training_configs_support_one_two_three_workers():
    import yaml
    from financeqa.modeling import DEFAULT_MODEL,DEFAULT_REVISION,chat_kwargs
    for path in sorted((ROOT/'configs').glob('*.yaml')):
        cfg=yaml.safe_load(path.read_text())
        assert cfg['model']['name']==DEFAULT_MODEL and cfg['model']['revision']==DEFAULT_REVISION,path
        assert chat_kwargs(cfg['model'])=={'enable_thinking':False}
        if 'training' in cfg:
            for workers in (1,2,3):
                validate_batch(cfg['training'],grpo='num_generations' in cfg['training'],workers=workers)
