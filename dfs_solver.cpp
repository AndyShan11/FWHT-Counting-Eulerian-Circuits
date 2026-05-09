/**
 * ====================================================================================
 * DFS-Based Eulerian Circuit Counter with Fleury Bridge Pruning
 * ====================================================================================
 *
 * Improved baseline for RQ2 comparison against the FWHT-based FPT method.
 *
 * Key optimizations over naive backtracking:
 *   1. Fleury pruning: At each step, Tarjan's O(V+E) bridge detection identifies
 *      edges whose removal disconnects the remaining graph.  Only non-bridge edges
 *      are explored; a bridge is taken only when it is the sole remaining edge.
 *   2. Short-circuit on forced moves: When rem_deg(u)==1, the move is forced and
 *      bridge detection is skipped entirely.
 *
 * Correctness guarantee:
 *   In a connected graph, taking a bridge when non-bridge alternatives exist always
 *   leads to an incomplete traversal (dead end).  Pruning bridge edges therefore
 *   preserves the exact Euler-circuit count.
 *
 * Final count:
 *   raw_count / d(start_vertex), where d(v) = degree of the starting vertex.
 *   Each undirected Euler circuit generates d(v) directed closed trails from v.
 *
 * Timeout behaviour:
 *   When the DFS exceeds the time limit, it records the partial raw count
 *   and elapsed time for extrapolation.
 *
 * Build:
 *   g++ -O3 -std=c++17 -funroll-loops dfs_solver.cpp -o dfs_solver
 *   # add -lstdc++fs if <filesystem> linking fails
 *
 * Usage:
 *   ./dfs_solver <directory_of_gr_files> [output_csv]
 *
 * ====================================================================================
 */

#pragma GCC optimize("O3,unroll-loops")

#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <functional>

namespace fs = std::filesystem;
using namespace std;

using CountT = long long;

class DFSEulerianCounter {
    struct Edge { int u, v, id; };
    int n, m;
    vector<vector<pair<int, int>>> adj;   // adj[u] = {(v, edge_id), ...}
    vector<Edge> edges;

    /* ---- Tarjan bridge detection (reused buffers) ---- */
    vector<int> disc, low;
    vector<bool> is_bridge;
    int bridge_timer;

    void bridgeDFS(int u, int parent_eid, const vector<bool>& used) {
        disc[u] = low[u] = bridge_timer++;
        for (auto& [v, eid] : adj[u]) {
            if (used[eid] || eid == parent_eid) continue;
            if (disc[v] == -1) {
                bridgeDFS(v, eid, used);
                low[u] = min(low[u], low[v]);
                if (low[v] > disc[u])
                    is_bridge[eid] = true;
            } else {
                low[u] = min(low[u], disc[v]);
            }
        }
    }

    void findBridges(const vector<bool>& used) {
        fill(disc.begin(), disc.end(), -1);
        fill(is_bridge.begin(), is_bridge.end(), false);
        bridge_timer = 0;
        for (int i = 0; i < n; i++) {
            if (disc[i] != -1) continue;
            bool has_unused = false;
            for (auto& [v, eid] : adj[i])
                if (!used[eid]) { has_unused = true; break; }
            if (has_unused) bridgeDFS(i, -1, used);
        }
    }

public:
    DFSEulerianCounter(int nodes)
        : n(nodes), m(0), adj(nodes), disc(nodes), low(nodes) {}

    void addEdge(int u, int v) {
        if (u < 0 || u >= n || v < 0 || v >= n) return;
        adj[u].push_back({v, m});
        adj[v].push_back({u, m});
        edges.push_back({u, v, m});
        m++;
    }

    int getN() const { return n; }
    int getM() const { return m; }

    int getGenus() const {
        if (m == 0) return 0;
        vector<bool> visited(n, false);
        int start = -1;
        for (int i = 0; i < n; ++i)
            if (!adj[i].empty()) { start = i; break; }
        if (start == -1) return 0;
        queue<int> q;
        q.push(start);
        visited[start] = true;
        int cnt = 1;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto& [v, eid] : adj[u])
                if (!visited[v]) { visited[v] = true; q.push(v); cnt++; }
        }
        return m - cnt + 1;
    }

    /**
     * Count Eulerian circuits by DFS backtracking with Fleury bridge pruning.
     *
     * @param timeout_sec     Wall-clock time limit in seconds.
     * @param out_timeout     Set to true if the search timed out.
     * @param out_partial_raw Raw closed-trail count found so far (before division).
     * @return                Exact circuit count on success, 0 on timeout.
     */
    CountT solve(double timeout_sec, bool& out_timeout, CountT& out_partial_raw) {
        out_timeout     = false;
        out_partial_raw = 0;

        cout << "  Nodes: " << n << ", Edges: " << m << endl;
        if (m == 0) return 0;

        // Eulerian condition: all degrees must be even
        for (int i = 0; i < n; ++i)
            if (adj[i].size() % 2 != 0) return 0;

        // Connectivity check among vertices with edges
        vector<bool> visited(n, false);
        int start_node = -1;
        for (int i = 0; i < n; ++i)
            if (!adj[i].empty()) { start_node = i; break; }
        if (start_node == -1) return 0;

        queue<int> bfs_q;
        bfs_q.push(start_node);
        visited[start_node] = true;
        while (!bfs_q.empty()) {
            int curr = bfs_q.front(); bfs_q.pop();
            for (auto& [v, eid] : adj[curr])
                if (!visited[v]) { visited[v] = true; bfs_q.push(v); }
        }
        for (int i = 0; i < n; ++i)
            if (!adj[i].empty() && !visited[i]) return 0;

        int genus = getGenus();
        int start_degree = (int)adj[start_node].size();
        cout << "  Circuit rank (g): " << genus
             << ", d(start)=" << start_degree
             << " [Fleury-DFS]" << endl;

        // Allocate bridge detection buffers
        is_bridge.resize(m);

        CountT raw_count = 0;
        vector<bool> used(m, false);
        vector<int> rem_deg(n, 0);
        for (int i = 0; i < n; i++) rem_deg[i] = (int)adj[i].size();

        auto start_time = chrono::high_resolution_clock::now();
        long long check_counter = 0;

        function<void(int, int)> dfs = [&](int u, int depth) {
            if (out_timeout) return;

            // Periodic timeout check (every 200K recursions)
            if (++check_counter % 200000 == 0) {
                auto now = chrono::high_resolution_clock::now();
                double elapsed = chrono::duration<double>(now - start_time).count();
                if (elapsed > timeout_sec) {
                    out_timeout = true;
                    return;
                }
            }

            if (depth == m) {
                if (u == start_node) raw_count++;
                return;
            }

            if (rem_deg[u] == 0) return;   // dead end

            /* ---- Forced move: only one edge left, skip bridge detection ---- */
            if (rem_deg[u] == 1) {
                for (auto& [v, eid] : adj[u]) {
                    if (!used[eid]) {
                        used[eid] = true; rem_deg[u]--; rem_deg[v]--;
                        dfs(v, depth + 1);
                        used[eid] = false; rem_deg[u]++; rem_deg[v]++;
                        return;
                    }
                }
                return;
            }

            /* ---- Bridge detection for branching vertices ---- */
            findBridges(used);

            // Partition edges from u into non-bridge / bridge
            // Use small inline vectors to avoid heap allocation
            pair<int,int> nb[64], br[64];
            int nb_cnt = 0, br_cnt = 0;
            for (auto& [v, eid] : adj[u]) {
                if (used[eid]) continue;
                if (is_bridge[eid])
                    br[br_cnt++] = {v, eid};
                else
                    nb[nb_cnt++] = {v, eid};
            }

            if (nb_cnt > 0) {
                // Branch on non-bridge edges only (bridges are provably dead ends)
                for (int i = 0; i < nb_cnt; i++) {
                    auto [v, eid] = nb[i];
                    used[eid] = true; rem_deg[u]--; rem_deg[v]--;
                    dfs(v, depth + 1);
                    if (out_timeout) return;
                    used[eid] = false; rem_deg[u]++; rem_deg[v]++;
                }
            } else if (br_cnt == 1) {
                // Forced bridge: sole remaining edge
                auto [v, eid] = br[0];
                used[eid] = true; rem_deg[u]--; rem_deg[v]--;
                dfs(v, depth + 1);
                used[eid] = false; rem_deg[u]++; rem_deg[v]++;
            }
            // else: multiple bridges from u => all paths disconnect => dead end
        };

        dfs(start_node, 0);
        out_partial_raw = raw_count;

        if (out_timeout) return 0;

        // Each undirected circuit yields d(start_node) directed trails from start
        CountT circuits = raw_count / start_degree;
        return circuits;
    }
};

/* ======================== DIMACS Graph Parser ======================== */
DFSEulerianCounter LoadGraph(const string& filename) {
    ifstream infile(filename);
    if (!infile.is_open())
        throw runtime_error("Cannot open graph file: " + filename);

    int n = 0;
    string line;
    while (getline(infile, line)) {
        if (line.empty() || line[0] == 'c') continue;
        stringstream ss(line);
        string token;
        ss >> token;
        if (token == "p") {
            string type;
            ss >> type >> n;
            break;
        }
    }
    DFSEulerianCounter solver(n);
    while (getline(infile, line)) {
        if (line.empty() || line[0] == 'c') continue;
        stringstream ss(line);
        string head;
        ss >> head;
        if (head == "e") {
            int u, v;
            ss >> u >> v;
            solver.addEdge(u - 1, v - 1);
        } else {
            try {
                int u = stoi(head), v;
                ss >> v;
                solver.addEdge(u - 1, v - 1);
            } catch (...) {}
        }
    }
    return solver;
}

/* ======================== Batch Processing Entry Point ======================== */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: dfs_solver <directory_of_gr_files> [output_csv]" << endl;
        return 1;
    }

    string folder_path = argv[1];
    string output_csv  = (argc >= 3) ? argv[2] : "results/eulerian_DFS_results.csv";

    ofstream csv_file(output_csv);
    if (!csv_file.is_open()) {
        cerr << "Error: Cannot open " << output_csv << " for writing." << endl;
        return 1;
    }
    csv_file << "Filename,N,M,Genus,Status,Circuits,Time(s),PartialRaw\n";

    const double TIMEOUT_SECONDS = 600.0;
    cout << ">>> DFS Solver (Fleury bridge pruning). Scanning: " << folder_path << endl;
    cout << ">>> Timeout: " << TIMEOUT_SECONDS << " s per graph" << endl;

    int success_count = 0, fail_count = 0, timeout_count = 0;

    // Collect and sort files by edge count (easy graphs first)
    struct FileInfo { fs::path path; int m_edges; };
    vector<FileInfo> files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".gr") {
            ifstream fin(entry.path().string());
            int file_m = 0;
            string ln;
            while (getline(fin, ln)) {
                if (!ln.empty() && ln[0] == 'p') {
                    stringstream ss(ln);
                    string tok, typ;
                    int nn;
                    ss >> tok >> typ >> nn >> file_m;
                    break;
                }
            }
            files.push_back({entry.path(), file_m});
        }
    }
    sort(files.begin(), files.end(),
         [](const FileInfo& a, const FileInfo& b) { return a.m_edges < b.m_edges; });

    for (const auto& finfo : files) {
        string filename  = finfo.path.filename().string();
        string full_path = finfo.path.string();

        cout << "\n----------------------------------------" << endl;
        cout << ">>> [Fleury-DFS] Processing: " << filename << endl;

        try {
            DFSEulerianCounter solver = LoadGraph(full_path);
            int graph_n = solver.getN();
            int graph_m = solver.getM();
            int graph_g = solver.getGenus();

            bool is_timeout = false;
            CountT partial_raw = 0;
            auto start = chrono::high_resolution_clock::now();
            CountT result = solver.solve(TIMEOUT_SECONDS, is_timeout, partial_raw);
            auto end = chrono::high_resolution_clock::now();

            double time_elapsed = chrono::duration<double>(end - start).count();

            if (is_timeout) {
                cout << "    [TIMEOUT] " << TIMEOUT_SECONDS << " s exceeded." << endl;
                cout << "    Partial raw trails: " << partial_raw << endl;
                csv_file << filename << ","
                         << graph_n << "," << graph_m << "," << graph_g << ","
                         << "TIMEOUT," << 0 << "," << time_elapsed << ","
                         << partial_raw << "\n";
                timeout_count++;
            } else {
                cout << "    Circuits : " << result << endl;
                cout << "    Time     : " << time_elapsed << " s" << endl;
                csv_file << filename << ","
                         << graph_n << "," << graph_m << "," << graph_g << ","
                         << "OK," << result << "," << time_elapsed << ","
                         << partial_raw << "\n";
                success_count++;
            }

        } catch (const exception& e) {
            cerr << "    [ERROR] " << filename << ": " << e.what() << endl;
            csv_file << filename << ",0,0,0,ERROR,0,0,0\n";
            fail_count++;
        }
    }

    csv_file.close();
    cout << "\n========================================" << endl;
    cout << ">>> DFS Batch complete. Success: " << success_count
         << " | Timeout: " << timeout_count
         << " | Failed: " << fail_count << endl;
    cout << ">>> Results: " << output_csv << endl;
    cout << "========================================" << endl;

    return 0;
}
