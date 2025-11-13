"""Quick test to verify task embeddings implementation."""

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0_config import Pi0Config
from openpi.models.model import Observation


def test_task_embeddings():
    """Test that task embeddings work correctly."""
    
    # Test 1: Task embeddings disabled (backward compatible)
    print("Test 1: Task embeddings disabled (num_tasks=0)")
    config_disabled = Pi0Config(num_tasks=0)
    model_disabled = config_disabled.create(jax.random.key(0))
    assert not hasattr(model_disabled, 'task_embeddings'), "Task embeddings should not exist when num_tasks=0"
    print("✓ Task embeddings disabled correctly")
    
    # Test 2: Task embeddings enabled
    print("\nTest 2: Task embeddings enabled (num_tasks=50)")
    config_enabled = Pi0Config(num_tasks=50, task_embedding_scale=1.0)
    model_enabled = config_enabled.create(jax.random.key(0))
    assert hasattr(model_enabled, 'task_embeddings'), "Task embeddings should exist when num_tasks=50"
    print("✓ Task embeddings created correctly")
    
    # Test 3: Forward pass with task_id
    print("\nTest 3: Forward pass with task_id")
    batch_size = 2
    obs_spec, action_spec = config_enabled.inputs_spec(batch_size=batch_size)
    
    # Create fake observation with task_id
    fake_obs = Observation(
        images={
            "base_0_rgb": jnp.ones((batch_size, 224, 224, 3)),
            "left_wrist_0_rgb": jnp.ones((batch_size, 224, 224, 3)),
            "right_wrist_0_rgb": jnp.ones((batch_size, 224, 224, 3)),
        },
        image_masks={
            "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
        },
        state=jnp.zeros((batch_size, config_enabled.action_dim)),
        task_id=jnp.array([5, 10], dtype=jnp.int32),  # Two different tasks
        tokenized_prompt=jnp.zeros((batch_size, config_enabled.max_token_len), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, config_enabled.max_token_len), dtype=jnp.bool_),
    )
    fake_actions = jnp.zeros((batch_size, config_enabled.action_horizon, config_enabled.action_dim))
    
    # Compute loss (this tests the full forward pass)
    rng = jax.random.key(42)
    loss = model_enabled.compute_loss(rng, fake_obs, fake_actions, train=False)
    assert loss.shape == (batch_size, config_enabled.action_horizon), f"Loss shape mismatch: {loss.shape}"
    print(f"✓ Forward pass successful, loss shape: {loss.shape}")
    
    # Test 4: Forward pass without task_id (should still work)
    print("\nTest 4: Forward pass without task_id (optional field)")
    fake_obs_no_task = Observation(
        images=fake_obs.images,
        image_masks=fake_obs.image_masks,
        state=fake_obs.state,
        task_id=None,  # No task ID
        tokenized_prompt=fake_obs.tokenized_prompt,
        tokenized_prompt_mask=fake_obs.tokenized_prompt_mask,
    )
    loss_no_task = model_enabled.compute_loss(rng, fake_obs_no_task, fake_actions, train=False)
    assert loss_no_task.shape == (batch_size, config_enabled.action_horizon)
    print(f"✓ Forward pass without task_id works, loss shape: {loss_no_task.shape}")
    
    # Test 5: Verify task embeddings affect output
    print("\nTest 5: Verify task embeddings affect output")
    fake_obs_task_a = fake_obs
    fake_obs_task_b = Observation(
        images=fake_obs.images,
        image_masks=fake_obs.image_masks,
        state=fake_obs.state,
        task_id=jnp.array([20, 30], dtype=jnp.int32),  # Different tasks
        tokenized_prompt=fake_obs.tokenized_prompt,
        tokenized_prompt_mask=fake_obs.tokenized_prompt_mask,
    )
    
    loss_task_a = model_enabled.compute_loss(rng, fake_obs_task_a, fake_actions, train=False)
    loss_task_b = model_enabled.compute_loss(rng, fake_obs_task_b, fake_actions, train=False)
    
    # Losses should be different when task IDs are different
    are_different = not jnp.allclose(loss_task_a, loss_task_b, atol=1e-6)
    assert are_different, "Task embeddings should affect the output!"
    print(f"✓ Task embeddings affect output (losses differ)")
    print(f"  Task [5,10] mean loss: {jnp.mean(loss_task_a):.4f}")
    print(f"  Task [20,30] mean loss: {jnp.mean(loss_task_b):.4f}")
    
    # Test 6: Check parameter count
    print("\nTest 6: Parameter count")
    from flax import nnx
    state = nnx.state(model_enabled)
    task_emb_params = state.task_embeddings.embedding.value
    num_params = np.prod(task_emb_params.shape)
    print(f"✓ Task embedding parameters: {num_params:,} ({task_emb_params.shape})")
    expected = 50 * 2048  # num_tasks * hidden_dim (for 300M action expert)
    print(f"  Expected: ~{expected:,} parameters")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nTask embeddings are ready to use!")
    print("Enable with: --model.num_tasks=50 --model.task_embedding_scale=1.0")


if __name__ == "__main__":
    test_task_embeddings()

