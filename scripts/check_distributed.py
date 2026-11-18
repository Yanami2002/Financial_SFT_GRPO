"""CPU/Gloo protocol probe: three processes, cross-rank GRPO groups, one writer. No model or training."""
import argparse
import json
from pathlib import Path
from financeqa.distributed import rank, world_size, gather_records
from train.grpo import DiagnosticReward


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    import torch.distributed as dist
    dist.init_process_group('gloo')
    try:
        if world_size()!=3:
            raise ValueError('This probe requires torchrun --nproc_per_node=3')
        if rank()==0:
            Path(a.output).mkdir(parents=True,exist_ok=True)
        dist.barrier()
        # Two groups of six; each rank holds four records, so groups cross rank boundaries.
        records=list(range(rank()*4,(rank()+1)*4))
        ids=[f'group-{i//6}' for i in records]
        outputs=['{"answer":20,"unit":"percent"}' if i%2==0 else '{}' for i in records]
        ref=[{'value':.2,'unit':'ratio'}]*4
        reward=DiagnosticReward(Path(a.output)/'reward_groups.jsonl',{'atol':1e-6,'rtol':.001},num_generations=6)
        values=reward(outputs,ref,ids)
        assert len(values)==4
        collected=gather_records([{'rank':rank(),'reward_count':len(values)}])
        dist.barrier()
        if rank()==0:
            groups=[json.loads(s) for s in (Path(a.output)/'reward_groups.jsonl').read_text().splitlines()]
            assert len(groups)==2 and all(g['n']==6 and g['accuracy']==.5 for g in groups)
            report={'world_size':world_size(),'cpu_only':True,'runs_training':False,
                    'cross_rank_groups':len(groups),'group_sizes':[g['n'] for g in groups],
                    'rank_results':collected,'single_writer_verified':True}
            (Path(a.output)/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(report,flush=True)
    finally:
        dist.destroy_process_group()

if __name__=='__main__':
    main()
