"""Verifiable DAPO outcome reward with a soft overlength penalty."""
from rewards.numerical import completion_text, score


def outcome(completion, reference, token_count, max_length, buffer_length, penalty_factor, tolerance):
    if not 0 < buffer_length < max_length or penalty_factor < 0:
        raise ValueError('Invalid DAPO overlength buffer or penalty')
    result = score(completion_text(completion), reference, **tolerance)
    penalty = -min(1.0, max(0.0, (token_count - (max_length - buffer_length)) / buffer_length))
    penalty *= penalty_factor
    return {**result, 'length_penalty': penalty,
            'reward': (1.0 if result['correct'] else -1.0) + penalty}


class DAPOReward:
    __name__ = 'dapo_outcome'

    def __init__(self, path, tolerance, num_generations, max_length, buffer_length, penalty_factor=1.0):
        from pathlib import Path
        from financeqa.distributed import is_main_process
        self.path = Path(path)
        self.tolerance = tolerance
        self.num_generations = num_generations
        self.max_length = max_length
        self.buffer_length = buffer_length
        self.penalty_factor = penalty_factor
        self.call = 0
        if is_main_process():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text('', encoding='utf-8')

    def assess(self, completion, reference, token_count):
        return outcome(completion, reference, token_count, self.max_length,
                       self.buffer_length, self.penalty_factor, self.tolerance)

    def __call__(self, completions, reference, id, completion_ids, **kwargs):
        import json
        import statistics
        from financeqa.distributed import gather_records, is_main_process
        from train.grpo import diagnostic_groups
        if not len(completions) == len(reference) == len(id) == len(completion_ids):
            raise ValueError('DAPO reward columns do not align')
        results = [self.assess(c, r, len(ids)) for c, r, ids in zip(completions, reference, completion_ids)]
        records = gather_records([{'id': key, 'score': result} for key, result in zip(id, results)])
        self.call += 1
        if is_main_process():
            with self.path.open('a', encoding='utf-8') as handle:
                for index, (key, values) in enumerate(diagnostic_groups(records, self.num_generations)):
                    rewards = [v['reward'] for v in values]
                    row = {'call': self.call, 'group': index, 'id': key, 'n': len(values),
                           'rewards': rewards, 'mean': statistics.mean(rewards),
                           'std': statistics.pstdev(rewards),
                           'accuracy': statistics.mean(float(v['correct']) for v in values),
                           'parseable_rate': statistics.mean(float(v['parseable']) for v in values)}
                    handle.write(json.dumps(row, allow_nan=False) + '\n')
        return [v['reward'] for v in results]
