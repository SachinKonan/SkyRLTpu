"""Citation prediction v2 utilities — recall-based scoring for multi-citation prediction."""

import re


def normalize_arxiv_id(s: str) -> str:
    """Normalize an arxiv ID by extracting the core YYMM.NNNNN pattern.

    Handles any prefix (arXiv:, cs:, [arxiv:...], URLs) and version suffixes.
    """
    s = s.strip()
    match = re.search(r"(\d{4}\.\d{4,5})", s)
    if match:
        return match.group(1)
    s = s.strip("[]")
    s = re.sub(r"^[A-Za-z-]+[:/]\s*", "", s)
    s = re.sub(r"v\d+$", "", s)
    return s.strip()


def extract_all_citations(text: str) -> set[str]:
    """Extract all arxiv IDs from <citation>...</citation> tags in text.

    Handles comma-separated IDs within a single tag and multiple tags.
    Returns a set of normalized arxiv IDs.
    """
    ids = set()
    for match in re.finditer(r"<citation>(.*?)</citation>", text, re.DOTALL):
        raw = match.group(1)
        for part in raw.split(","):
            part = part.strip()
            if part:
                normalized = normalize_arxiv_id(part)
                if normalized:
                    ids.add(normalized)
    return ids


def compute_recall_reward(text: str, ground_truth_ids: list[str], max_ratio: float = 2.0) -> float:
    """Compute recall-based reward with spam penalty.

    Args:
        text: Full chat history concatenated as a string.
        ground_truth_ids: List of ground-truth arxiv IDs for this subsection.
        max_ratio: If |predicted| > max_ratio * |ground_truth|, reward = 0 (spam penalty).

    Returns:
        recall = |predicted & ground_truth| / |ground_truth|, or 0.0 if spam penalty triggered.
    """
    predicted = extract_all_citations(text)
    if not ground_truth_ids:
        return 0.0

    gt_set = {normalize_arxiv_id(gid) for gid in ground_truth_ids}

    # Spam penalty: too many predictions relative to ground truth
    if len(predicted) > max_ratio * len(gt_set):
        return 0.0

    correct = predicted & gt_set
    return len(correct) / len(gt_set)
