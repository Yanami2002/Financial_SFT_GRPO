import pytest
from rewards.numerical import parse_output, score, numerical_reward
from rewards.format_reward import format_reward


@pytest.mark.parametrize('text,value,unit', [
    ('{"answer":"20%","unit":"percent"}', .2, 'ratio'),
    ('{"answer":0.2,"unit":"ratio"}', .2, 'ratio'),
    ('{"answer":"(1,250.50)","unit":"usd"}', -1250.5, 'usd'),
    ('{"answer":"−3.5","unit":"percent"}', -.035, 'ratio'),
    ('{"answer":2,"unit":"usd_million"}', 2e6, 'usd'),
    ('{"answer":"1e-3","unit":"ratio"}', .001, 'ratio'),
])
def test_parse(text, value, unit):
    result = parse_output(text)
    assert result['parseable'] and result['value'] == pytest.approx(value) and result['unit'] == unit


@pytest.mark.parametrize('text', ['bad', '```json\n{"answer":2,"unit":"ratio"}\n```', '{}',
    '{"unit":"ratio"}', '{"answer":true,"unit":"ratio"}', '{"answer":null,"unit":"ratio"}',
    '{"answer":NaN,"unit":"ratio"}', '{"answer":Infinity,"unit":"ratio"}',
    '{"answer":"1e9999","unit":"ratio"}', '{"answer":2}', '{"answer":2,"unit":[]}',
    '{"answer":"20%","unit":"ratio"}', '{"answer":2,"answer":3,"unit":"ratio"}',
    '{"answer":"12,34","unit":"count"}', '{"answer":"1 or 2","unit":"ratio"}', '[]'])
def test_bad_outputs(text):
    assert not parse_output(text)['parseable']


def test_tolerance_sign_and_units():
    ref = {'value': .2, 'unit': 'ratio', 'atol': 0}
    assert score('{"answer":20.01,"unit":"percent"}', ref, rtol=.001)['correct']
    assert not score('{"answer":20,"unit":"ratio"}', ref)['correct']
    assert not score('{"answer":-20,"unit":"percent"}', ref)['correct']
    assert not score('{"answer":0.2,"unit":"usd"}', ref)['correct']
    assert score('{"answer":0.20001,"unit":"ratio"}', ref, atol=.0001, rtol=0)['correct']
    assert not score('{"answer":0.20001,"unit":"ratio"}', ref, atol=.000001, rtol=0)['correct']
    assert not score('{"answer":0.2,"unit":"ratio"}', None)['scorable']
    with pytest.raises(ValueError):
        score('{}', ref, atol=-1)


def test_reward_callbacks_and_unscorable():
    cs = [[{'role':'assistant','content':'{"answer":20,"unit":"percent"}'}], '{}']
    refs = [{'value':.2,'unit':'ratio'}, {'value':.2,'unit':'ratio'}]
    assert numerical_reward(cs, refs) == [1., 0.]
    assert format_reward(cs) == [1., 0.]
    assert numerical_reward([cs[0]], [None]) == [0.]
    with pytest.raises(ValueError):
        numerical_reward(cs, [])


def test_reference_precision_and_invalid_reference():
    assert score('{"answer":20,"unit":"percent"}', {'value':'.2','unit':'ratio'})['correct']
    assert score('{"answer":20,"unit":"percent"}', {'value':.2004,'unit':'ratio','atol':.0005}, rtol=0)['correct']
    assert not score('{"answer":20,"unit":"percent"}', {'value':.2006,'unit':'ratio','atol':.0005}, rtol=0)['correct']
    assert not score('{"answer":20,"unit":"percent"}', {'value':.2,'unit':[]})['scorable']
    assert not score('{"answer":20,"unit":"percent"}', {'value':float('nan'),'unit':'ratio'})['scorable']


def test_explicit_percent_symbol_unit_alias():
    ref = {'value':.2,'unit':'ratio'}
    assert score('{"answer":20,"unit":"%"}',ref)['correct']
    assert score('{"answer":"20%","unit":"%"}',ref)['correct']
    assert not score('{"answer":.2,"unit":"%"}',ref)['parseable']  # invalid JSON .2
    assert not score('{"answer":0.2,"unit":"%"}',ref)['correct']
