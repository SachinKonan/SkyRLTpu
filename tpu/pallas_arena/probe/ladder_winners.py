"""Verbatim PASSING kernels from the prompt-ladder run (job 3687041).

Not fixtures and not scaffolding: these are exactly what gemma-4-31B-it wrote,
byte for byte, extracted from the run JSONL. Kept because a report that claims
"working code" and shows none is not evidence, and because these are the
seeds a future RL run would start from.

  rg_lru  | rung p1 | reward 1.0000 | gate `all`
  flce    | rung p1 | reward 0.4990 | gate `all`
"""

# ---------------------------------------------------------------- rg_lru
# GATE all | rg_lru | PASS reward=1.0000
# probe-4x2048x2560: cand 2.199ms vs ref 2.201ms (1.001x)
# probe-2x1024x2560: cand 0.286ms vs ref 0.285ms (0.998x)
# probe-holdout-2x1500x2560: cand 0.790ms vs ref 0.806ms (1.019x)
# peak HBM 30.30GB
RG_LRU_P1_REWARD = 1.0
RG_LRU_P1 = r"""
import jax
import jax.numpy as jnp

def kernel(x, a, reset):
    """
    Implements a diagonal linear scan recurrence: h_t = a_t * h_{t-1} + gx_t.
    
    Args:
        x: [b, t, d] bfloat16 input
        a: [b, t, d] float32 gates
        reset: [b, t] bool reset mask
    
    Returns:
        h: [b, t, d] float32 hidden states
    """
    
    # Pre-process inputs to match the oracle precision and recurrence form
    # Cast a and reset to float32, force a_t to 0 wherever reset_t is True
    # a32 shape: [b, t, d]
    a32 = a.astype(jnp.float32) * (1.0 - reset[..., None].astype(jnp.float32))
    
    # Precompute gx = sqrt(max(1 - a^2, 0)) * x
    # Cast x to float32 for precision during the accumulation process
    # gx shape: [b, t, d]
    gx = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a32), 0.0)) * x.astype(jnp.float32)
    
    # associative_scan processes along the first axis. 
    # We transpose time (t) to the lead dimension.
    # Shapes become [t, b, d]
    a_t = jnp.transpose(a32, (1, 0, 2))
    gx_t = jnp.transpose(gx, (1, 0, 2))
    
    def combine(lhs, rhs):
        """
        Associative operator for linear recurrence.
        The state transforms as: h_t = a_t * h_{t-1} + gx_t
        This corresponds to the operator: (a_prev, h_prev) ⊕ (a_curr, gx_curr) = (a_curr * a_prev, a_curr * h_prev + gx_curr)
        """
        a_accum, h_accum = lhs
        a_curr, gx_curr = rhs
        
        # New cumulative multiplicative factor
        next_a = a_curr * a_accum
        # New cumulative hidden state (the actual recurrence)
        next_h = a_curr * h_accum + gx_curr
        
        return next_a, next_h

    # jax.lax.associative_scan efficiently computes the prefix sum (scan)
    # utilizing a log-time parallel algorithm on TPUs.
    # Input is a tuple (pytree) of the values to scan.
    # Result is a tuple of the same structure as input.
    _, h_t = jax.lax.associative_scan(combine, (a_t, gx_t), axis=0)
    
    # Transpose the result back to original dimensions [b, t, d]
    h = jnp.transpose(h_t, (1, 0, 2))
    
    return h
"""

# ------------------------------------------------------------------ flce
# GATE all | flce | PASS reward=0.4990
# probe-4096x2880x151936: cand 15.650ms vs ref 6.487ms (0.415x)
# probe-2048x2880x151936: cand 6.793ms vs ref 4.037ms (0.594x)
# probe-holdout-3000x2880x151936: cand 12.325ms vs ref 6.535ms (0.530x)
# peak HBM 30.35GB
FLCE_P1_REWARD = 0.49900779416750735
FLCE_P1 = r"""
import jax
import jax.numpy as jnp

@jax.custom_vjp
def kernel(hidden, w, targets):
    """
    Computes log-probabilities without materializing the full [n, v] logits array.
    Tiling is used along the token axis 'n' to maintain a small HBM footprint.
    """
    n, h = hidden.shape
    v = w.shape[1]
    T = 256  # Tiling size for the token axis
    
    # Handle cases where n is not a multiple of T by padding
    padded_n = (n + T - 1) // T * T
    padding = padded_n - n
    h_padded = jnp.pad(hidden, ((0, padding), (0, 0)))
    t_padded = jnp.pad(targets, (0, padding), constant_values=0)
    
    h_tiles = h_padded.reshape(-1, T, h)
    t_tiles = t_padded.reshape(-1, T)
    
    def scan_body(carry, x):
        h_tile, t_tile = x
        # Ensure float32 accumulation for precision
        logits = jnp.matmul(h_tile.astype(jnp.float32), w.astype(jnp.float32))
        # Extract logits for the target indices: logits[i, targets[i]]
        tl = jnp.take_along_axis(logits, t_tile[:, None], axis=1)[:, 0]
        # Log-sum-exp across the vocabulary dimension
        lse = jax.nn.logsumexp(logits, axis=-1)
        return carry, tl - lse
    
    _, logprobs_tiles = jax.lax.scan(scan_body, None, (h_tiles, t_tiles))
    logprobs = logprobs_tiles.reshape(-1)
    # Return only the non-padded portion
    return logprobs[:n].astype(jnp.float32)

def kernel_fwd(hidden, w, targets):
    """Forward pass for custom_vjp; returns output and residuals for backward pass."""
    return kernel(hidden, w, targets), (hidden, w, targets)

def kernel_bwd(res, G):
    """
    Backward pass for custom_vjp.
    Recomputes logits per tile to avoid materializing the large [n, v] array.
    G is the gradient of the loss with respect to the log-probabilities.
    """
    hidden, w, targets = res
    n, h = hidden.shape
    v = w.shape[1]
    T = 256
    
    padded_n = (n + T - 1) // T * T
    padding = padded_n - n
    
    h_padded = jnp.pad(hidden, ((0, padding), (0, 0)))
    t_padded = jnp.pad(targets, (0, padding), constant_values=0)
    G_padded = jnp.pad(G, (0, padding))
    
    h_tiles = h_padded.reshape(-1, T, h)
    t_tiles = t_padded.reshape(-1, T)
    G_tiles = G_padded.reshape(-1, T)
    
    def scan_body(carry, x):
        h_tile, t_tile, G_tile = x
        # Recompute logits for the backward pass
        logits = jnp.matmul(h_tile.astype(jnp.float32), w.astype(jnp.float32))
        probs = jax.nn.softmax(logits, axis=-1)
        
        # Derivative of log(softmax(L)_target) w.r.t. L_j is: 
        # delta(j, target) - softmax(L)_j
        # We scale this by G_tile (gradient of loss w.r.t log-prob)
        grad_logits = -G_tile[:, None] * probs
        # Add the one-hot component for the target indices
        grad_logits = grad_logits.at[jnp.arange(T), t_tile].add(G_tile)
        
        # Gradient w.r.t hidden: (grad_logits) @ w.T
        grad_h_tile = jnp.matmul(grad_logits, w.astype(jnp.float32).T)
        return carry, grad_h_tile
    
    _, grad_h_tiles = jax.lax.scan(scan_body, None, (h_tiles, t_tiles, G_tiles))
    grad_h = grad_h_tiles.reshape(-1, h)
    
    # Return gradients for (hidden, w, targets); w is frozen, targets is discrete
    return (grad_h[:n].astype(hidden.dtype), None, None)

# Register the VJP functions
kernel.defvjp(kernel_fwd, kernel_bwd)
"""
