"""Fail closed on missing native-control evidence before training sees a sample."""
import math


def validate_choice(choice, budget, max_tokens):
    tokens = choice.get("token_ids")
    mask = choice.get("loss_mask")
    audit = choice.get("thinking_budget") or {}
    positions = choice.get("forced_token_positions")
    lps = (choice.get("logprobs") or {}).get("token_logprobs")
    if not isinstance(tokens, list) or len(tokens) > max_tokens:
        raise ValueError("native completion has invalid token IDs/length")
    if not isinstance(mask, list) or len(mask) != len(tokens) or any(type(x) not in (int, float) or x not in (0, 1) for x in mask):
        raise ValueError("native completion has missing/invalid loss mask")
    if positions != [i for i, x in enumerate(mask) if x == 0]:
        raise ValueError("native completion forced positions disagree with mask")
    if audit.get("enforced") is not True or audit.get("budget_basis") != "phase1_generated_tokens":
        raise ValueError("native completion missing enforcement audit")
    counted = audit.get("counted_phase1_tokens")
    if type(counted) is not int or not 0 <= counted <= budget or bool(positions) != audit.get("forced"):
        raise ValueError("native completion exceeded budget or has inconsistent audit")
    if not isinstance(lps, list) or len(lps) != len(tokens) or any(x is None or not math.isfinite(x) for x in lps):
        raise ValueError("native completion missing finite behavior logprobs")
    # Same synthetic logprob convention as TwoPhaseTokenCompleter. Sampled
    # tokens retain their original model logprobs; forced tokens have weight 0.
    choice["logprobs"]["token_logprobs"] = [lp if m else 0.0 for lp, m in zip(lps, mask)]
