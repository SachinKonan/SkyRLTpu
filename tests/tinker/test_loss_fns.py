import jax
import jax.numpy as jnp
import numpy as np

from skyrl.tinker.loss_fns import LossFnConfig, cispo_loss, compute_per_token_losses, reduce_per_sequence_losses


def test_cispo_loss_clipping_and_masking():
    target_logprobs = jnp.array([0.0, 5.0, -5.0], dtype=jnp.float32)
    sampling_logprobs = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    advantages = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    loss_mask = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)
    loss_fn_config = LossFnConfig(
        clip_low_threshold=jnp.array(0.8, dtype=jnp.float32),
        clip_high_threshold=jnp.array(1.2, dtype=jnp.float32),
        tis_imp_ratio_cap=jnp.array(-1.0, dtype=jnp.float32),
        old_logprobs_from_target=jnp.array(0.0, dtype=jnp.float32),
        token_mean=jnp.array(0.0, dtype=jnp.float32),
    )

    actual = np.asarray(cispo_loss(target_logprobs, loss_mask, sampling_logprobs, advantages, loss_fn_config))

    ratios = np.exp(np.asarray(target_logprobs - sampling_logprobs))
    clipped_ratios = np.clip(ratios, 0.8, 1.2)
    expected = -(clipped_ratios * np.asarray(target_logprobs) * np.asarray(advantages))
    expected[2] = 0.0

    assert np.allclose(actual, expected)


def test_cispo_stops_gradient_through_clipped_ratio():
    sampling_logprobs = jnp.array([0.1, -0.2], dtype=jnp.float32)
    advantages = jnp.array([1.3, -0.7], dtype=jnp.float32)
    loss_mask = jnp.ones((2,), dtype=jnp.float32)
    loss_fn_config = LossFnConfig(
        clip_low_threshold=jnp.array(0.8, dtype=jnp.float32),
        clip_high_threshold=jnp.array(1.2, dtype=jnp.float32),
        tis_imp_ratio_cap=jnp.array(-1.0, dtype=jnp.float32),
        old_logprobs_from_target=jnp.array(0.0, dtype=jnp.float32),
        token_mean=jnp.array(0.0, dtype=jnp.float32),
    )

    def total_loss(target_logprobs):
        return cispo_loss(target_logprobs, loss_mask, sampling_logprobs, advantages, loss_fn_config).sum()

    target_logprobs = jnp.array([0.0, 0.2], dtype=jnp.float32)
    grad = np.asarray(jax.grad(total_loss)(target_logprobs))

    clipped_ratio = np.clip(
        np.exp(np.asarray(target_logprobs - sampling_logprobs)),
        0.8,
        1.2,
    )
    expected_grad = -(clipped_ratio * np.asarray(advantages))

    assert np.allclose(grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_compute_per_token_losses_dispatches_per_sequence():
    target_logprobs = jnp.array([[-0.2, -0.4], [-0.1, -0.3]], dtype=jnp.float32)
    mask = jnp.ones_like(target_logprobs)
    sampling_logprobs = jnp.zeros_like(target_logprobs)
    advantages = jnp.ones_like(target_logprobs)
    config = LossFnConfig(
        clip_low_threshold=jnp.array([0.8, 0.8], dtype=jnp.float32),
        clip_high_threshold=jnp.array([1.2, 1.2], dtype=jnp.float32),
        tis_imp_ratio_cap=jnp.array([-1.0, -1.0], dtype=jnp.float32),
        old_logprobs_from_target=jnp.array([0.0, 0.0], dtype=jnp.float32),
        token_mean=jnp.array([0.0, 0.0], dtype=jnp.float32),
    )

    actual = compute_per_token_losses(
        jnp.array([0, 2], dtype=jnp.int32),
        target_logprobs,
        mask,
        sampling_logprobs,
        sampling_logprobs,
        advantages,
        config,
    )

    expected_ce = -np.asarray(target_logprobs[0])
    expected_ppo = -np.minimum(np.exp(np.asarray(target_logprobs[1])), 1.2)
    np.testing.assert_allclose(np.asarray(actual[0]), expected_ce)
    np.testing.assert_allclose(np.asarray(actual[1]), expected_ppo)


def test_cispo_with_target_old_logprobs_and_token_tis():
    target = jnp.array([[0.0, np.log(2.0)]], dtype=jnp.float32)
    rollout = jnp.zeros_like(target)
    losses = compute_per_token_losses(
        jnp.array([3], dtype=jnp.int32),
        target,
        jnp.ones_like(target),
        jnp.full_like(target, -99.0),
        rollout,
        jnp.ones_like(target),
        LossFnConfig(
            clip_low_threshold=jnp.array([1.0], dtype=jnp.float32),
            clip_high_threshold=jnp.array([6.0], dtype=jnp.float32),
            tis_imp_ratio_cap=jnp.array([1.5], dtype=jnp.float32),
            old_logprobs_from_target=jnp.array([1.0], dtype=jnp.float32),
            token_mean=jnp.array([1.0], dtype=jnp.float32),
        ),
    )

    # CISPO current/old is exactly one. TIS old/rollout is [1, 2], capped to [1, 1.5].
    np.testing.assert_allclose(np.asarray(losses), np.asarray([[-0.0, -1.5 * np.log(2.0)]]), rtol=1e-5)


def test_token_mean_reduction_preserves_global_batch_weighting():
    losses = jnp.array([[0.25, 0.0], [0.25, 0.5]], dtype=jnp.float32)
    weights = jnp.array([[0.5, 0.0], [0.5, 1.0]], dtype=jnp.float32)
    config = LossFnConfig(
        clip_low_threshold=jnp.array([0.8, 0.8], dtype=jnp.float32),
        clip_high_threshold=jnp.array([1.2, 1.2], dtype=jnp.float32),
        tis_imp_ratio_cap=jnp.array([-1.0, -1.0], dtype=jnp.float32),
        old_logprobs_from_target=jnp.array([0.0, 0.0], dtype=jnp.float32),
        token_mean=jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    np.testing.assert_allclose(np.asarray(reduce_per_sequence_losses(losses, weights, config)), [0.25, 0.75])
