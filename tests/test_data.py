import json
import pytest
from data.prepare_data import convert, reference_for, prepare
from financeqa.common import read_jsonl, training_rows


def sample(id='report-0'):
    return {'id': id, 'pre_text': ['Revenue rose from 100 to 120.'], 'post_text': ['All amounts are USD.'],
            'table': [['year','revenue'],['2020','100'],['2021','120']],
            'qa': {'question':'What is the percentage increase?', 'answer':'20%', 'exe_ans':.2,
                   'program':'subtract(120, 100), divide(#0, 100)', 'gold_inds': {'text_0':'SECRET_GOLD'}}}


def test_official_fields_and_no_leakage():
    r = convert(sample(), 'train', 'revision')
    assert r['id'] == 'report-0' and r['table'][0] == ['year','revenue']
    assert 'Revenue rose' in r['prompt'][1]['content']
    rendered = json.dumps(r['prompt'])
    assert 'SECRET_GOLD' not in rendered and 'subtract(' not in rendered
    assert '"answer": 0.2' not in rendered  # generic schema examples are fixed, not gold
    assert r['program'].startswith('subtract') and r['reference']['unit'] == 'ratio'


@pytest.mark.parametrize('answer,exe,reason', [
    ('20%', .2, None), ('20%', 20, None), ('-20%', -.2, None),
    ('20%', -.2, 'annotation_execution_disagreement'),
    ('20%', .5, 'annotation_execution_disagreement'), ('', .2, 'missing_written_answer'),
    ('20', 20, 'ambiguous_unit'), ('$20 million', 20, 'ambiguous_unit'),
    ('yes', 'yes', 'nonnumeric_execution')])
def test_reference_filter(answer, exe, reason):
    qa = sample()['qa']; qa.update(answer=answer, exe_ans=exe)
    ref, got = reference_for(qa)
    assert got == reason
    assert (ref is None) == (reason is not None)


def test_prepare_official_splits_and_stats(tmp_path):
    raw, out = tmp_path/'raw', tmp_path/'out'; raw.mkdir()
    for split in ('train','dev','test'):
        row = sample(split+'-0')
        row['pre_text'].append('Report split fixture: ' + split)
        (raw/f'{split}.json').write_text(json.dumps([row]))
    stats = prepare(raw, out)
    assert all(s['total'] == 1 and s['verified_numeric'] == 1 for s in stats.values())
    train = read_jsonl(out/'train_grpo.jsonl')
    assert train[0]['split'] == 'train' and 'reference' in train[0]
    assert 'reference' not in json.dumps(train[0]['prompt'])
    assert not (out/'test_grpo.jsonl').exists()
    sft = read_jsonl(out/'train_sft.jsonl')[0]
    assert sft['messages'][-1]['role'] == 'assistant'
    assert training_rows(out/'train_sft.jsonl')
    with pytest.raises(ValueError, match='exclusively'):
        training_rows(out/'dev_sft.jsonl')
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['provenance'] == 'local_hash_only'


def test_cross_split_duplicates_stop_preparation(tmp_path):
    for split in ('train','dev','test'):
        (tmp_path/f'{split}.json').write_text(json.dumps([sample('same-id')]))
    with pytest.raises(ValueError, match='Duplicate'):
        prepare(tmp_path, tmp_path/'out')
    assert not (tmp_path/'out'/'train_sft.jsonl').exists()
