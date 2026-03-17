/**
 * ======================================================================================
 * PARALLEL OPTIMIZED ALGORITHM: Exact Eulerian Circuit Counting via Twisted Adjacency & FWHT
 * PARADIGM: Fixed-Parameter Tractable (FPT) parameterized by Genus (g)
 * OPTIMIZATIONS: 1D Flat Matrix, Gray Code Transitions, Zero-Allocation, OpenMP Multi-threading, Atomic Timeout
 * ======================================================================================
 */

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include <iostream>
#include <vector>
#include <queue>
#include <cassert>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <omp.h> // Import OpenMP parallel library
#include <filesystem>
#include <atomic> // Import atomic library for thread-safe timeout flag

namespace fs = std::filesystem;
using namespace std;

// ====================== 0. Precision Engine ======================
#ifdef USE_GMP
    #include <gmpxx.h>
    using CountT = mpz_class;
#else
    using CountT = __int128_t;
    std::ostream& operator<<(std::ostream& os, __int128_t n) {
        if (n == 0) return os << "0";
        if (n < 0) { os << "-"; n = -n; }
        string s;
        while (n > 0) { s += (char)('0' + (n % 10)); n /= 10; }
        reverse(s.begin(), s.end());
        return os << s;
    }
#endif

// ====================== 1. High-Performance 1D Flat Matrix Module ======================
struct FlatMatrix {
    int n;
    vector<CountT> mat;

    FlatMatrix(int size) : n(size), mat(size * size, 0) {}

    inline void clear_and_set_identity() {
        fill(mat.begin(), mat.end(), 0);
        for (int i = 0; i < n; ++i) mat[i * n + i] = 1;
    }

    inline void multiply(const FlatMatrix& other, FlatMatrix& res) const {
        fill(res.mat.begin(), res.mat.end(), 0);
        const CountT* A = mat.data();
        const CountT* B = other.mat.data();
        CountT* C = res.mat.data();
        
        for (int i = 0; i < n; ++i) {
            int i_n = i * n;
            for (int k = 0; k < n; ++k) {
                CountT val = A[i_n + k];
                if (!val) continue; 
                int k_n = k * n;
                for (int j = 0; j < n; ++j) {
                    C[i_n + j] += val * B[k_n + j];
                }
            }
        }
    }
};

CountT compute_trace_power(const FlatMatrix& W_start, int k, FlatMatrix& M_res, FlatMatrix& M_base, FlatMatrix& M_tmp) {
    int n = W_start.n;
    M_res.clear_and_set_identity();
    M_base.mat = W_start.mat; 

    FlatMatrix* cur_res = &M_res;
    FlatMatrix* cur_base = &M_base;
    FlatMatrix* swap_tmp = &M_tmp;

    while (k > 0) {
        if (k & 1) {
            cur_res->multiply(*cur_base, *swap_tmp);
            swap(cur_res, swap_tmp);
        }
        if (k > 1) {
            cur_base->multiply(*cur_base, *swap_tmp);
            swap(cur_base, swap_tmp);
        }
        k >>= 1;
    }

    CountT sum = 0;
    for (int i = 0; i < n; ++i) sum += cur_res->mat[i * n + i];
    return sum;
}

// ====================== 2. Fast Walsh-Hadamard Transform (FWHT) ======================
void FWHT(vector<CountT>& a) {
    int n = a.size();
    for (int len = 1; 2 * len <= n; len <<= 1) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                CountT u = a[i + j];
                CountT v = a[i + len + j];
                a[i + j] = u + v;
                a[i + len + j] = u - v;
            }
        }
    }
}

// ====================== 3. Graph Structure and Solver ======================
class EulerianCounter {
    struct Edge { int u, v, id; };
    int n, m;
    vector<vector<Edge>> adj;
    vector<Edge> edges;

public:
    EulerianCounter(int nodes) : n(nodes), m(0), adj(nodes) {}

    void addEdge(int u, int v) {
        if (u < 0 || u >= n || v < 0 || v >= n) return;
        adj[u].push_back({u, v, m});
        adj[v].push_back({v, u, m});
        edges.push_back({u, v, m});
        m++;
    }

    // Add timeout_sec and out_timeout parameters
    CountT solve(double timeout_sec, bool& out_timeout) {
        out_timeout = false;
        
        cout << "Debug -> Number of nodes read n: " << n << ", Number of edges read m: " << m << endl;
        
        if (m == 0) return 0;
        for (int i = 0; i < n; ++i) {
            if (adj[i].size() % 2 != 0) return 0;
        }

        vector<bool> visited(n, false);
        vector<bool> is_tree_edge(m, false);
        queue<int> q;
        
        int start_node = 0;
        while (start_node < n && adj[start_node].empty()) start_node++;
        if (start_node == n) return 0; 

        q.push(start_node);
        visited[start_node] = true;
        int edge_connected_nodes = 1;

        while (!q.empty()) {
            int curr = q.front();
            q.pop();
            for (const auto& edge : adj[curr]) {
                if (!visited[edge.v]) {
                    visited[edge.v] = true;
                    is_tree_edge[edge.id] = true;
                    q.push(edge.v);
                    edge_connected_nodes++;
                }
            }
        }

        int g = m - edge_connected_nodes + 1; 
        
        int max_threads = omp_get_max_threads();
        cout << "[Info] Nodes: " << n << " | Edges: " << m << " | Genus(g): " << g << endl;
        cout << "[Info] OpenMP Engine Active. Utilizing " << max_threads << " threads." << endl;

        if (g > 30) throw runtime_error("Genus exceeds 30. O(2^g) computation will take geological time.");

        vector<int> cotree_edges;
        for (int i = 0; i < m; ++i) {
            if (!is_tree_edge[i]) cotree_edges.push_back(i);
        }

        FlatMatrix W_base(2 * m);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                int u1 = edges[i].u, v1 = edges[i].v;
                int u2 = edges[j].u, v2 = edges[j].v;
                if (v1 == u2 && i != j) W_base.mat[i * W_base.n + j] = 1;
                if (v1 == v2 && i != j) W_base.mat[i * W_base.n + j + m] = 1;
                if (u1 == u2 && i != j) W_base.mat[(i + m) * W_base.n + j] = 1;
                if (u1 == v2 && i != j) W_base.mat[(i + m) * W_base.n + j + m] = 1;
            }
        }

        int num_states = 1 << g;
        vector<CountT> trace_array(num_states, 0);

        // --- New: Timeout control variables ---
        double start_wtime = omp_get_wtime();
        std::atomic<bool> timeout_flag{false};

        // Core parallel region: Distribute state computation
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            
            FlatMatrix W_local = W_base;
            FlatMatrix m_res(2 * m), m_base_pow(2 * m), m_tmp(2 * m);

            if (0 % num_threads == thread_id) {
                trace_array[0] = compute_trace_power(W_local, m, m_res, m_base_pow, m_tmp);
            }

            for (int i = 1; i < num_states; ++i) {
                // Skip subsequent computations if timeout is detected
                if (timeout_flag.load(std::memory_order_relaxed)) continue;

                int flip_bit = __builtin_ctz(i); 
                int edge_id = cotree_edges[flip_bit];
                
                for (int r = 0; r < 2 * m; ++r) {
                    W_local.mat[r * W_local.n + edge_id] = -W_local.mat[r * W_local.n + edge_id];
                    W_local.mat[r * W_local.n + edge_id + m] = -W_local.mat[r * W_local.n + edge_id + m];
                }
                
                int gray_val = i ^ (i >> 1); 
                
                if (gray_val % num_threads == thread_id) {
                    // Check for timeout
                    if (omp_get_wtime() - start_wtime > timeout_sec) {
                        timeout_flag.store(true, std::memory_order_relaxed);
                    } else {
                        trace_array[gray_val] = compute_trace_power(W_local, m, m_res, m_base_pow, m_tmp);
                    }
                }
            }
        }

        // Check status after exiting parallel region
        if (timeout_flag.load()) {
            out_timeout = true;
            return 0;
        }

        FWHT(trace_array);
        CountT eulerian_circuits = trace_array[num_states - 1] / num_states / (2 * m);
        return eulerian_circuits;
    }
};

// ====================== 4. File Parsing Module ======================
EulerianCounter LoadGraph(const string& filename) {
    ifstream infile(filename);
    if (!infile.is_open()) throw runtime_error("Cannot open graph file.");
    
    int n = 0; string line, type;
    while (getline(infile, line)) {
        if (line.empty() || line[0] == 'c') continue;
        stringstream ss(line); string token; ss >> token;
        if (token == "p") { ss >> type >> n; break; } 
    }
    EulerianCounter solver(n);
    while (getline(infile, line)) {
        if (line.empty() || line[0] == 'c') continue;
        stringstream ss(line); string head; ss >> head;
        if (head == "e") {
            int u, v; ss >> u >> v; solver.addEdge(u - 1, v - 1);
        } else {
            try { int u = stoi(head); int v; ss >> v; solver.addEdge(u - 1, v - 1); } catch (...) {}
        }
    }
    return solver;
}

// ====================== 5. Batch Processing and Command-Line Entry ======================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: ./euler_omp <directory_containing_gr_files>" << endl;
        return 1;
    }

    string folder_path = argv[1];
    string output_csv = "D:\\cpp\\.vscode\\eulerian_RQ1_results.csv";

    ofstream csv_file(output_csv);
    if (!csv_file.is_open()) {
        cerr << "Error: Cannot open " << output_csv << " for writing." << endl;
        return 1;
    }
    csv_file << "Filename,Eulerian Circuits,Time Elapsed (s)\n";

    // Set timeout to 600 seconds (10 minutes)
    const double TIMEOUT_SECONDS = 600.0;
    cout << ">>> Engine Initialized. Scanning folder: " << folder_path << "..." << endl;
    cout << ">>> TIMEOUT set to " << TIMEOUT_SECONDS << " seconds." << endl;

    int success_count = 0;
    int fail_count = 0;
    int timeout_count = 0;

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".gr") {
            string filename = entry.path().filename().string();
            string full_path = entry.path().string();

            cout << "\n----------------------------------------" << endl;
            cout << ">>> Processing: " << filename << endl;

            try {
                EulerianCounter solver = LoadGraph(full_path);
                
                bool is_timeout = false;
                auto start = chrono::high_resolution_clock::now();
                // Pass timeout duration and status flag
                CountT result = solver.solve(TIMEOUT_SECONDS, is_timeout);
                auto end = chrono::high_resolution_clock::now();
                
                double time_elapsed = chrono::duration<double>(end - start).count();
                
                if (is_timeout) {
                    cout << "    [WARNING] TIMEOUT EXCEEDED (" << TIMEOUT_SECONDS << "s)!" << endl;
                    csv_file << filename << ",TIMEOUT,>600.0\n";
                    timeout_count++;
                } else {
                    cout << "    Circuits : " << result << endl;
                    cout << "    Time     : " << time_elapsed << " s" << endl;
                    csv_file << filename << "," << result << "," << time_elapsed << "\n";
                    success_count++;
                }

            } catch (const exception& e) {
                cerr << "    [ERROR] Skipped " << filename << ": " << e.what() << endl;
                csv_file << filename << ",ERROR: " << e.what() << ",N/A\n";
                fail_count++;
            }
        }
    }

    csv_file.close();
    cout << "\n========================================" << endl;
    cout << ">>> Batch processing complete!" << endl;
    cout << ">>> Success: " << success_count << " | Timeout: " << timeout_count << " | Failed: " << fail_count << endl;
    cout << ">>> All results saved to " << output_csv << " (Open with Excel)" << endl;
    cout << "========================================" << endl;

    return 0;
}