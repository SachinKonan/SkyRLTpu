"""Assemble compact code-agent messages; optionally include DEVELOPMENT reward.

Usage: python scripts/create_agent_prompt.py --previous results/trial/reward.json
Do not point --previous at a locked holdout or official leaderboard response.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(previous=None):
    base=(ROOT/'prompts/SYSTEM_V2.md').read_text()+'\n\n'+(ROOT/'prompts/TASK_V2.md').read_text()
    if previous:
        report=json.loads(Path(previous).read_text())
        if not report.get('reward_is_proxy_not_official',False):
            raise ValueError('Only DEVELOPMENT proxy-reward files may be inserted')
        items=sorted(report.get('per_family',{}).items(),key=lambda kv:kv[1])
        summary={'development_proxy_reward':report.get('reward'),
                 'per_shift_development_scores':items,
                 'warning':'These proxy categories are diagnostic only. Do not hardcode them.'}
        base+='\n\n# Last DEVELOPMENT feedback (not hidden FAIR test)\n'+json.dumps(summary,indent=2)
        base+='\n\n'+(ROOT/'prompts/ITERATE_V2.md').read_text()
    return base


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--previous',help='Development reward.json ONLY')
    p.add_argument('--out',help='Optional output text file')
    a=p.parse_args()
    s=build(a.previous)
    if a.out: Path(a.out).write_text(s)
    else: print(s)

if __name__=='__main__':main()
