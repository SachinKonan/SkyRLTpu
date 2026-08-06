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