You are an expert competitive programmer and algorithm engineer.
Solve the following open-scoring optimization problem from the Frontier-CS benchmark.

## Problem statement

Title: Job Shop Scheduling (JSPLIB-style) — Open Optimization Track

Overview
--------
You are given a classic Job Shop Scheduling Problem (JSSP). There are J jobs and M machines.
Each job must be processed exactly once on each machine, in a job-specific order (its *route*).
Processing is non-preemptive. A machine can process at most one operation at a time. The goal is to minimize the *makespan*:
the completion time of the last operation among all jobs.

This problem is NP-hard. We therefore use an *open scoring* scheme that rewards better (lower) makespans. See **Scoring** below.

Terminology
-----------
• Operation: A single (job, machine) processing step with a fixed processing time.  
• Route of a job j: A sequence of M distinct machines (0..M-1) listing the order in which job j must visit them.  
• Precedence (job chain): If the k-th operation of job j precedes its (k+1)-th, the latter cannot start before the former finishes.  
• Resource constraint (machine): Operations assigned to the same machine cannot overlap in time.  
• Makespan (C_max): The maximum completion time over all operations in the schedule.

Input Format
------------
The input is plain text with 0-based indices.

Line 1:
  J M
  • J (integer): number of jobs (J ≥ 1)
  • M (integer): number of machines (M ≥ 1)

Lines 2..(J+1): one line per job j in order j = 0..J-1. Each line contains 2*M integers
representing the route and processing times for job j:

  m_0 p_0  m_1 p_1  ...  m_{M-1} p_{M-1}

where:
  • m_k ∈ {0,1,...,M-1} is the machine index of the k-th operation of job j.  
  • p_k is a positive integer (processing time of that operation).  
  • Each machine index must appear **exactly once** in a job’s line (every job uses every machine exactly once).  
  • The order of the pairs on the line determines the job’s precedence constraints.

Output Format
-------------
You must output **exactly M lines**.
Line m (for m = 0..M-1) must contain **J distinct integers**: a permutation of {0,1,...,J-1}.  
This permutation specifies the order in which machine m processes the J jobs (from first to last).

Important:
• You **do not** print start or finish times.  
• Your permutations must mention each job exactly once on every machine. Otherwise, the checker will reject the output.  
• The judge constructs the earliest-feasible schedule implied by your machine orders and the job precedence constraints
  (equivalently: the longest-path length in the disjunctive graph with your chosen orientations on machine arcs).

Validity Rules
--------------
Your output is rejected if any of the following occurs:
• A machine line does not contain a permutation of {0..J-1} (duplicate/missing job index; out-of-range index).  
• The machine orders together with the job precedence constraints induce a cycle in the disjunctive graph
  (i.e., there exists no feasible schedule consistent with your machine orders).

Scoring (Lower is Better)
-------------------------
Let P be the makespan computed from your output for a test case. The answer file for each test contains two integers:
(B, T). For this problem, **T is fixed to 0** (a trivial lower bound), and **B > 0** is the makespan of a simple feasible
baseline schedule (a naïve dispatch heuristic). The checker applies the general formula:

  If B ≤ T:   score = 1.0 if P ≤ T else 0.0
  Else:       score = clamp( (B - P) / (B - T), 0, 1 )

With T = 0 and B > 0, this simplifies to:

  score = clamp( 1 - P / B, 0, 1 )

Your final problem score is the average of your per-test scores. The checker prints partial credit messages containing
the substring “Ratio: <value>” as required by the judge.

Constraints
-----------
The official test set is deliberately challenging:
• Sizes range up to ~ (J, M) ≈ (50, 25) (total operations up to ~1,250).  
• Processing times are positive integers and may range broadly (including very large values).  
• Route structures include random, nearly-flow, block-flow, and strong bottlenecks (one or two machines dominating).

Tips
----
Feasible and competitive schedules often come from combinations of:
• Priority rules (SPT/LPT/weighted), bottleneck-aware dispatching.  
• Local improvement via adjacent swaps per machine.  
• Metaheuristics (tabu search, simulated annealing, iterated local search).  
• Shifting bottleneck heuristics or relax-and-fix styles.

Example (Illustrative Only; Not in the Tests)
---------------------------------------------
Input:
  3 2
  0 3  1 4
  1 2  0 5
  0 4  1 1

Valid Output (two lines; each line is a permutation of {0,1,2}):
  2 0 1
  1 2 0

This tells the judge to process jobs on machine 0 in order [2,0,1] and on machine 1 in order [1,2,0].
The judge then computes the earliest-feasible schedule and its makespan.


## Judge contract
- Submit ONE C++17 (gnu++17) translation unit. It is compiled with:
  `g++ main.cpp -O2 -pipe -std=gnu++17 -o a`
- Your program reads the test case from standard input and writes its answer to
  standard output. No file or network IO. No arguments.
- Per test case: CPU time limit 1s (wall 2s),
  memory limit 512 MB.
- You are scored on 1 test case(s); the official checker assigns each case
  a continuous ratio (higher is better). Your score is the mean ratio across cases.
  A ratio of 1.0 corresponds to the judge's reference quality; on some problems it is
  possible to exceed 1.0 by beating the reference. Invalid output scores 0 on that case.
- Exploit the full per-case time limit for search/optimization, but never exceed it —
  a timed-out case scores 0.

You are iteratively optimizing mean checker ratio.
No previous code available.
Current mean checker ratio (higher is better): 0.000000
Target: 1.0. Current gap: 1.000000. Further improvements will also be generously rewarded.

Reason about how you could further improve on the previous approach.
Ideally, try to do something different than the above algorithm. Could be using
different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your
hyperparameters, etc. Unless you make a meaningful improvement, you will not be
rewarded.

Rules:
- Output exactly one complete C++ program in a single ```cpp code block.
- Make the program deterministic or internally seeded (fixed seed).
- Include a short comment at the top summarizing your algorithm.

## Starting program to improve on
This program's PRODUCTION score is 0.073942. HIGHER scores are BETTER (maximize). Your goal is a program that scores meaningfully higher than 0.073942 at the same production budget.

```cpp
/*
 * Job‑Shop Open‑Optimization Solver
 * ---------------------------------
 * The program builds a feasible schedule by a greedy list‑scheduling (a
 * Giffler‑Thompson style heuristic).  At every step it schedules the
 * operation that can start the earliest; ties are broken by the larger
 * amount of remaining work (sum of processing times of the not yet
 * processed operations of that job).  Random tie‑breaking is used and the
 * heuristic is run repeatedly within the time limit; the best schedule
 * (smallest makespan) is kept and its machine orders are output.
 *
 * The schedule itself is constructed step by step, therefore the
 * resulting machine orders are guaranteed to be feasible (they come from
 * an actual schedule, so no cycles can appear).
 *
 * Complexity per iteration: O(J·M) operations (≤ 1250) – easily fits the
 * time limit, allowing many repetitions for better solutions.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int J, M;
    if (!(cin >> J >> M)) return 0;

    vector<vector<int>> jobMach(J, vector<int>(M));
    vector<vector<long long>> jobProc(J, vector<long long>(M));
    for (int j = 0; j < J; ++j) {
        for (int k = 0; k < M; ++k) {
            int m; long long p;
            cin >> m >> p;
            jobMach[j][k] = m;
            jobProc[j][k] = p;
        }
    }

    // suffix[j][i] = sum_{t=i}^{M-1} proc of job j (remaining work incl. current)
    vector<vector<long long>> suffix(J, vector<long long>(M + 1, 0));
    for (int j = 0; j < J; ++j) {
        suffix[j][M] = 0;
        for (int i = M - 1; i >= 0; --i)
            suffix[j][i] = suffix[j][i + 1] + jobProc[j][i];
    }

    // Greedy schedule generator (returns makespan and machine orders)
    auto runSchedule = [&](mt19937& rng) -> pair<long long, vector<vector<int>>> {
        vector<int> opIdx(J, 0);                 // next operation index per job
        vector<long long> ready(J, 0);           // completion time of last scheduled op of each job
        vector<long long> machAvail(M, 0);       // when each machine becomes free
        vector<vector<int>> order(M);
        for (int m = 0; m < M; ++m) order[m].reserve(J);

        const int totalOps = J * M;
        int scheduled = 0;
        while (scheduled < totalOps) {
            long long bestStart = LLONG_MAX;
            // 1) find the earliest possible start time among all ready operations
            for (int j = 0; j < J; ++j) {
                int idx = opIdx[j];
                if (idx == M) continue;
                int m = jobMach[j][idx];
                long long start = max(ready[j], machAvail[m]);
                if (start < bestStart) bestStart = start;
            }
            // 2) among those with earliest start, keep the ones with maximal remaining work
            long long maxRem = -1;
            for (int j = 0; j < J; ++j) {
                int idx = opIdx[j];
                if (idx == M) continue;
                int m = jobMach[j][idx];
                long long start = max(ready[j], machAvail[m]);
                if (start == bestStart) {
                    long long rem = suffix[j][idx];
                    if (rem > maxRem) maxRem = rem;
                }
            }
            // 3) collect candidates and pick one uniformly at random
            vector<int> cand;
            cand.reserve(J);
            for (int j = 0; j < J; ++j) {
                int idx = opIdx[j];
                if (idx == M) continue;
                int m = jobMach[j][idx];
                long long start = max(ready[j], machAvail[m]);
                if (start == bestStart && suffix[j][idx] == maxRem)
                    cand.push_back(j);
            }
            uniform_int_distribution<int> dist(0, (int)cand.size() - 1);
            int chosen = cand[dist(rng)];

            // schedule the chosen operation
            int idx = opIdx[chosen];
            int m = jobMach[chosen][idx];
            long long p = jobProc[chosen][idx];
            long long start = bestStart;
            long long finish = start + p;

            ready[chosen] = finish;
            machAvail[m] = finish;
            order[m].push_back(chosen);
            ++opIdx[chosen];
            ++scheduled;
        }

        long long makespan = 0;
        for (long long t : ready) if (t > makespan) makespan = t;
        return {makespan, order};
    };

    // deterministic first run (still uses the same random choices)
    mt19937 rng(123456789);
    long long bestMakespan = LLONG_MAX;
    vector<vector<int>> bestOrder;

    {
        auto res = runSchedule(rng);
        bestMakespan = res.first;
        bestOrder = move(res.second);
    }

    // repeat while we have time left
    auto tStart = chrono::steady_clock::now();
    const double TIME_LIMIT = 0.94; // seconds
    while (true) {
        auto now = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(now - tStart).count();
        if (elapsed > TIME_LIMIT) break;
        auto res = runSchedule(rng);
        if (res.first < bestMakespan) {
            bestMakespan = res.first;
            bestOrder = move(res.second);
        }
    }

    // output machine orders
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < J; ++i) {
            if (i) cout << ' ';
            cout << bestOrder[m][i];
        }
        cout << '\n';
    }
    return 0;
}
```
