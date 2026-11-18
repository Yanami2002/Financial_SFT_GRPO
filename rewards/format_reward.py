from rewards.numerical import completion_text, parse_output


def format_reward(completions, **kwargs):
    return [float(parse_output(completion_text(c))["parseable"]) for c in completions]
