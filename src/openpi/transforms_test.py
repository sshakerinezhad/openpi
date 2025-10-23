import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_proprio_dropout_whole():
    # Test dropping the entire proprioceptive state
    transform = _transforms.ProprioDropout(dropout_whole_proprio_pct=1.0)

    data = {"state": np.array([1.0, 2.0, 3.0, 4.0])}
    transformed = transform(data)

    assert "proprio_visibility_mask" in transformed
    assert np.all(transformed["proprio_visibility_mask"] == 0)
    assert transformed["proprio_visibility_mask"].shape == data["state"].shape


def test_proprio_dropout_no_dropout():
    # Test with zero dropout probability
    transform = _transforms.ProprioDropout(dropout_whole_proprio_pct=0.0)

    data = {"state": np.array([1.0, 2.0, 3.0, 4.0])}
    transformed = transform(data)

    # Should be a no-op
    assert transformed is data


def test_proprio_dropout_groups():
    # Test dropping specific groups of dimensions
    # We'll set probability to 1.0 to make it deterministic
    transform = _transforms.ProprioDropout(
        dropout_whole_proprio_pct=0.0,
        proprio_groups=[
            ([0, 1], 1.0),  # Drop first two dimensions with 100% probability
            ([3], 1.0),     # Drop fourth dimension with 100% probability
        ]
    )

    data = {"state": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
    transformed = transform(data)

    assert "proprio_visibility_mask" in transformed
    expected_mask = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    assert np.all(transformed["proprio_visibility_mask"] == expected_mask)


def test_proprio_dropout_no_state():
    # Test that it handles missing state gracefully
    transform = _transforms.ProprioDropout(dropout_whole_proprio_pct=1.0)

    data = {"actions": np.array([[1, 2, 3]])}
    transformed = transform(data)

    # Should return data unchanged
    assert "proprio_visibility_mask" not in transformed
    assert transformed is data


def test_proprio_dropout_groups_no_dropout():
    # Test groups with zero probability
    transform = _transforms.ProprioDropout(
        dropout_whole_proprio_pct=0.0,
        proprio_groups=[
            ([0, 1], 0.0),  # Never drop
        ]
    )

    data = {"state": np.array([1.0, 2.0, 3.0, 4.0])}

    # Run multiple times to verify it's consistently all ones
    for _ in range(10):
        transformed = transform(data)
        assert "proprio_visibility_mask" in transformed
        assert np.all(transformed["proprio_visibility_mask"] == 1.0)


def test_tokenize_prompt_with_discrete_state():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    transform = _transforms.TokenizePrompt(tokenizer, discrete_state_input=True)

    data = {
        "prompt": "Pick up the cup",
        "state": np.array([0.5, -0.3, 0.8]),
    }

    transformed = transform(data)

    assert "tokenized_prompt" in transformed
    assert "tokenized_prompt_mask" in transformed
    # Prompt should be removed
    assert "prompt" not in transformed


def test_tokenize_prompt_with_discrete_state_and_mask():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    transform = _transforms.TokenizePrompt(tokenizer, discrete_state_input=True)

    data = {
        "prompt": "Pick up the cup",
        "state": np.array([0.5, -0.3, 0.8, 0.1]),
        "proprio_visibility_mask": np.array([1.0, 0.0, 1.0, 0.0]),  # Mask out 2nd and 4th dims
    }

    transformed = transform(data)

    assert "tokenized_prompt" in transformed
    assert "tokenized_prompt_mask" in transformed
    # The proprio_visibility_mask should be passed to the tokenizer
    # We can verify by checking that the tokenization worked without errors


def test_tokenize_prompt_discrete_state_missing():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    transform = _transforms.TokenizePrompt(tokenizer, discrete_state_input=True)

    data = {"prompt": "Pick up the cup"}

    with pytest.raises(ValueError, match="State is required"):
        transform(data)


def test_paligemma_tokenizer_with_mask():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)

    state = np.array([0.5, -0.3, 0.8, 0.1])
    mask = np.array([1.0, 0.0, 1.0, 0.0])

    tokens, token_mask = tokenizer.tokenize("test prompt", state, mask)

    assert tokens.shape[0] == 64
    assert token_mask.shape[0] == 64
    # Verify that tokens were created (basic sanity check)
    assert np.any(token_mask)


def test_paligemma_tokenizer_with_state_no_mask():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)

    state = np.array([0.5, -0.3, 0.8])

    tokens, token_mask = tokenizer.tokenize("test prompt", state, None)

    assert tokens.shape[0] == 64
    assert token_mask.shape[0] == 64
    # Verify that tokens were created
    assert np.any(token_mask)


def test_proprio_dropout_pipeline():
    # Integration test: ProprioDropout -> TokenizePrompt
    # This tests the full pipeline to ensure the visibility mask is properly propagated

    dropout_transform = _transforms.ProprioDropout(
        dropout_whole_proprio_pct=0.0,
        proprio_groups=[
            ([1, 2], 1.0),  # Always drop dimensions 1 and 2
        ]
    )

    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    tokenize_transform = _transforms.TokenizePrompt(tokenizer, discrete_state_input=True)

    # Apply ProprioDropout
    data = {
        "prompt": "Move forward",
        "state": np.array([0.1, 0.2, 0.3, 0.4]),
    }
    data = dropout_transform(data)

    # Verify the mask was created
    assert "proprio_visibility_mask" in data
    expected_mask = np.array([1.0, 0.0, 0.0, 1.0])
    assert np.all(data["proprio_visibility_mask"] == expected_mask)

    # Apply TokenizePrompt
    data = tokenize_transform(data)

    # Verify tokenization succeeded
    assert "tokenized_prompt" in data
    assert "tokenized_prompt_mask" in data
    assert "prompt" not in data  # Should be removed

    # State and mask should still be in the data
    assert "state" in data
    assert "proprio_visibility_mask" in data


def test_proprio_dropout_with_multidim_state():
    # Test with multi-dimensional state (e.g., sequence of states)
    transform = _transforms.ProprioDropout(
        dropout_whole_proprio_pct=0.0,
        proprio_groups=[
            ([0], 1.0),  # Drop first dimension
        ]
    )

    data = {"state": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    transformed = transform(data)

    assert "proprio_visibility_mask" in transformed
    expected_mask = np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    assert np.all(transformed["proprio_visibility_mask"] == expected_mask)
