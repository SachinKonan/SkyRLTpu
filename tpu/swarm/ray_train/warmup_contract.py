"""Pure shape validation shared by the controller and trainer warmup."""

def shapes(sequence_length, token_budget, row_shard, *, uniform=0, buckets=()):
    if min(sequence_length, token_budget, row_shard) <= 0:
        raise ValueError("warmup requires positive length, budget and row shard")
    lengths = [uniform] if uniform else sorted(set(buckets))
    if not lengths or any(n <= 0 or n > sequence_length for n in lengths):
        raise ValueError("warmup requires bounded, explicit production sequence buckets")
    if not uniform and max(lengths) != sequence_length:
        raise ValueError("warmup buckets must include the maximum training length")
    result = []
    for length in lengths:
        # The packer always admits a singleton, even if its padded row exceeds
        # the token budget. Match the resulting FSDP padding in that case.
        rows = max(row_shard, (token_budget // length // row_shard) * row_shard)
        result.append((rows, length))
    return result
