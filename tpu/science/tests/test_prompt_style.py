from tpu.science import training_env

ENV = dict(SCIENCE_PLACEMENT_BACKEND='cpu', SCIENCE_PLACEMENT_HELPER='fast_proxy_v1',
           SCIENCE_PLACEMENT_RUNTIME='cpu300-4g-v1', SCIENCE_PLACEMENT_PROMPT='ibm17-v3')


def test_standard_style_adds_persona_framing_and_nudge(monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('SCIENCE_PROMPT_STYLE', 'standard')
    text = training_env.candidate_prompt('placement', 'def place(): pass', '{"reward": 0.5}')
    assert text.startswith(training_env.PERSONA['placement'] + '\n\n')
    assert training_env.OPTIMIZING['placement'] + '\nHere is the last code we ran:' in text
    assert text.rstrip().endswith('Unless you make a meaningful improvement, you will not be rewarded.')
    assert 'Previous candidate:' not in text and 'Improve this candidate.' not in text


def test_legacy_style_is_unchanged(monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('SCIENCE_PROMPT_STYLE', raising=False)
    text = training_env.candidate_prompt('placement', 'def place(): pass', '{"reward": 0.5}')
    assert not text.startswith(training_env.PERSONA['placement'])
    assert 'Previous candidate:' in text and text.rstrip().endswith('Improve this candidate.')


def test_repair_prompts_keep_the_repair_instruction(monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('SCIENCE_PROMPT_STYLE', 'standard')
    text = training_env.candidate_prompt('placement', 'def place(): pass', 'boom', repair=True)
    assert 'Repair this invalid program' in text
