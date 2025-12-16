#include <iostream>
#include <vector>
#include <map>
#include <mpi.h>
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>

using namespace std;

// --- CONFIGURATION ---
const int CUTOFF = 128; 
const int MPI_RECURSION_LIMIT = 512; 
const int TAG_WORK = 0;
const int TAG_RESULT = 1;
const int TAG_MEM_UPDATE = 4;
const int TAG_TERMINATE = 99;

// --- MEMORY SYSTEM ---
// Keep SmartBuffer for the big MPI messages to avoid frequent re-allocation
struct SmartBuffer {
    float* data;
    size_t capacity; 
    SmartBuffer() : data(nullptr), capacity(0) {}
    ~SmartBuffer() { if (data) delete[] data; }
    void ensure_size(int n) {
        size_t needed = (size_t)n * n;
        if (needed > capacity) {
            if (data) delete[] data;
            capacity = needed; 
            data = new float[capacity];
        }
    }
};

// --- GLOBAL STATE ---
struct TaskState {
    int count;             
    float* results[7];       
    int n;                 
    int parent_rank;       
    TaskState() {
        count = 0;
        for (int i = 0; i < 7; i++) results[i] = nullptr;
        n = 0; parent_rank = -1;
    }
};

std::map<long, TaskState> ongoing_tasks;
std::vector<long> node_memory; 
long my_current_memory = 0;    

// --- HELPERS ---
void log_debug(int rank, string msg) {
   // printf("[DEBUG] Rank %d: %s\n", rank, msg.c_str());
}

float* allocate_empty(int n) { return new float[n * n]; } // removed () to skip zero-init speedup

void verify_result(const float* A, const float* B, const float* C, int n) {
    cout << "--- Verifying Result (Checking 10 random spots) ---" << endl;
    bool pass = true;
    for(int k=0; k<10; k++) {
        int r = rand() % n;
        int c = rand() % n;
        float sum = 0.0f;
        for(int i=0; i<n; i++) {
            sum += A[r * n + i] * B[i * n + c];
            cout << A[r * n + i] << " * " << B[i * n + c] << " = " << A[r * n + i] * B[i * n + c] << endl;
        }
        cout << "Sum: " << sum << ", C[" << r << "][" << c << "] = " << C[r * n + c] << endl;
        float diff = fabs(sum - C[r * n + c]);
        // Strassen error can grow slightly, so we use a loose epsilon
        if (diff > 0.1f) { 
            cout << "[FAIL] At (" << r << "," << c << ") Expected: " << sum << " Got: " << C[r*n+c] << endl;
            pass = false;
        } else {
            // cout << "[OK] (" << r << "," << c << ") Matches." << endl;
        }
    }
    if (pass) cout << "SUCCESS: Result verification passed!" << endl;
    else cout << "FAILURE: Result verification failed." << endl;
}

void free_m(float* m) { if (m) delete[] m; }

// --- MATH KERNELS ---
void add(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) C[i] = A[i] + B[i];
}

void sub(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) C[i] = A[i] - B[i];
}

void multiply_std(const float* A, const float* B, float* C, int n) {
    // We must zero-init C here because we accumulate into it
    memset(C, 0, n * n * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float temp = A[i * n + k];
            for (int j = 0; j < n; j++) C[i * n + j] += temp * B[k * n + j];
        }
    }
}

// --- LOCAL STRASSEN (RESTORED PARALLEL TASKS) ---
void local_strassen_recursive(float* A, float* B, float* C, int n) {
    if (n <= CUTOFF) {
        multiply_std(A, B, C, n);
        return;
    }

    int k = n / 2;
    // We stick to 'new' here to allow OpenMP tasks to run safely without complex arena locking
    float *a11 = new float[k*k], *a12 = new float[k*k], *a21 = new float[k*k], *a22 = new float[k*k];
    float *b11 = new float[k*k], *b12 = new float[k*k], *b21 = new float[k*k], *b22 = new float[k*k];
    float *p1 = new float[k*k], *p2 = new float[k*k], *p3 = new float[k*k], *p4 = new float[k*k];
    float *p5 = new float[k*k], *p6 = new float[k*k], *p7 = new float[k*k];

    // Decompose (Parallel Copy)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            a11[i*k+j] = A[i*n+j];       a12[i*k+j] = A[i*n+(j+k)];
            a21[i*k+j] = A[(i+k)*n+j];   a22[i*k+j] = A[(i+k)*n+(j+k)];
            b11[i*k+j] = B[i*n+j];       b12[i*k+j] = B[i*n+(j+k)];
            b21[i*k+j] = B[(i+k)*n+j];   b22[i*k+j] = B[(i+k)*n+(j+k)];
        }
    }

    // Parallel Tasks for Recursion
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            { float *tA = new float[k*k], *tB = new float[k*k]; add(a11, a22, tA, k); add(b11, b22, tB, k); local_strassen_recursive(tA, tB, p1, k); delete[] tA; delete[] tB; }
            #pragma omp task
            { float *tA = new float[k*k]; add(a21, a22, tA, k); local_strassen_recursive(tA, b11, p2, k); delete[] tA; }
            #pragma omp task
            { float *tB = new float[k*k]; sub(b12, b22, tB, k); local_strassen_recursive(a11, tB, p3, k); delete[] tB; }
            #pragma omp task
            { float *tB = new float[k*k]; sub(b21, b11, tB, k); local_strassen_recursive(a22, tB, p4, k); delete[] tB; }
            #pragma omp task
            { float *tA = new float[k*k]; add(a11, a12, tA, k); local_strassen_recursive(tA, b22, p5, k); delete[] tA; }
            #pragma omp task
            { float *tA = new float[k*k], *tB = new float[k*k]; sub(a21, a11, tA, k); add(b11, b12, tB, k); local_strassen_recursive(tA, tB, p6, k); delete[] tA; delete[] tB; }
            #pragma omp task
            { float *tA = new float[k*k], *tB = new float[k*k]; sub(a12, a22, tA, k); add(b21, b22, tB, k); local_strassen_recursive(tA, tB, p7, k); delete[] tA; delete[] tB; }
        }
    }

    // Assembly
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            C[i*n+j]         = p1[i*k+j] + p4[i*k+j] - p5[i*k+j] + p7[i*k+j];
            C[i*n+(j+k)]     = p3[i*k+j] + p5[i*k+j];
            C[(i+k)*n+j]     = p2[i*k+j] + p4[i*k+j];
            C[(i+k)*n+(j+k)] = p1[i*k+j] - p2[i*k+j] + p3[i*k+j] + p6[i*k+j];
        }
    }

    delete[] a11; delete[] a12; delete[] a21; delete[] a22;
    delete[] b11; delete[] b12; delete[] b21; delete[] b22;
    delete[] p1; delete[] p2; delete[] p3; delete[] p4; delete[] p5; delete[] p6; delete[] p7;
}

// --- NETWORK LOGIC ---
void broadcast_memory_update(int my_rank) {
    if (my_rank < node_memory.size()) node_memory[my_rank] = my_current_memory;
    long payload[2];
    payload[0] = (long)my_rank;
    payload[1] = my_current_memory;
    if (my_rank > 0) {
        MPI_Bsend(payload, 2, MPI_LONG, 0, TAG_MEM_UPDATE, MPI_COMM_WORLD);
    }
}

std::vector<int> find_best_workers(int num_procs, int my_rank) {
    std::vector<std::pair<long, int>> sorted_nodes;
    for (int i = 1; i < num_procs; i++) {
        if (i != my_rank) sorted_nodes.push_back({node_memory[i], i});
    }
    std::random_shuffle(sorted_nodes.begin(), sorted_nodes.end());
    std::stable_sort(sorted_nodes.begin(), sorted_nodes.end(), 
        [](const std::pair<long, int>& a, const std::pair<long, int>& b) { return a.first < b.first; });

    std::vector<int> targets;
    if (sorted_nodes.empty()) return targets;
    int needed = 7;
    int available = sorted_nodes.size();
    for (int i = 0; i < std::min(needed, available); i++) targets.push_back(sorted_nodes[i].second);
    if (targets.size() > 0) {
        while (targets.size() < 7) targets.push_back(targets[targets.size() % available]);
    }
    return targets;
}

void assemble_and_send(long matrix_id, TaskState& state, int my_rank) {
    int n = state.n;
    int k = n / 2;
    float* C = new float[n*n]; 
    float* p[7];
    for (int i = 0; i < 7; i++) p[i] = state.results[i];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            C[i*n+j]         = p[0][i*k+j] + p[3][i*k+j] - p[4][i*k+j] + p[6][i*k+j];
            C[i*n+(j+k)]     = p[2][i*k+j] + p[4][i*k+j];
            C[(i+k)*n+j]     = p[1][i*k+j] + p[3][i*k+j];
            C[(i+k)*n+(j+k)] = p[0][i*k+j] - p[1][i*k+j] + p[2][i*k+j] + p[5][i*k+j];
        }
    }

    MPI_Bsend(&matrix_id, 1, MPI_LONG, state.parent_rank, TAG_RESULT, MPI_COMM_WORLD);
    MPI_Bsend(&n, 1, MPI_INT, state.parent_rank, TAG_RESULT, MPI_COMM_WORLD);
    MPI_Bsend(C, n * n, MPI_FLOAT, state.parent_rank, TAG_RESULT, MPI_COMM_WORLD);

    delete[] C;
    for (int i = 0; i < 7; i++) free_m(state.results[i]);
    ongoing_tasks.erase(matrix_id);

    long freed = (7 * k * k) + (n * n);
    if (freed > my_current_memory) my_current_memory = 0; else my_current_memory -= freed;
    broadcast_memory_update(my_rank);
}

void handle_new_work(int source, long matrix_id, int n, float* A, float* B, int my_rank, int num_procs) {
    my_current_memory += (2L * n * n);
    broadcast_memory_update(my_rank);

    std::vector<int> targets = find_best_workers(num_procs, my_rank);
    if (targets.size() < 7 || n < MPI_RECURSION_LIMIT) {
        float* C = new float[n*n];
        local_strassen_recursive(A, B, C, n); 
        
        MPI_Bsend(&matrix_id, 1, MPI_LONG, source, TAG_RESULT, MPI_COMM_WORLD);
        MPI_Bsend(&n, 1, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD);
        MPI_Bsend(C, n * n, MPI_FLOAT, source, TAG_RESULT, MPI_COMM_WORLD);
        
        delete[] C;
        my_current_memory -= (2L * n * n);
        broadcast_memory_update(my_rank);
        return;
    }

    TaskState state;
    state.n = n;
    state.parent_rank = source;
    ongoing_tasks[matrix_id] = state;

    int k = n / 2;
    float *a11 = new float[k*k], *a12 = new float[k*k], *a21 = new float[k*k], *a22 = new float[k*k];
    float *b11 = new float[k*k], *b12 = new float[k*k], *b21 = new float[k*k], *b22 = new float[k*k];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            a11[i * k + j] = A[i * n + j];       a12[i * k + j] = A[i * n + (j + k)];
            a21[i * k + j] = A[(i + k) * n + j]; a22[i * k + j] = A[(i + k) * n + (j + k)];
            b11[i * k + j] = B[i * n + j];       b12[i * k + j] = B[i * n + (j + k)];
            b21[i * k + j] = B[(i + k) * n + j]; b22[i * k + j] = B[(i + k) * n + (j + k)];
        }
    }

    for (int i = 0; i < 7; i++) {
        int target_rank = targets[i];
        long child_id = (matrix_id * 7) + (i + 1);
        
        float *TA = new float[k*k], *TB = new float[k*k];
        if (i == 0) { add(a11, a22, TA, k); add(b11, b22, TB, k); }      
        else if (i == 1) { add(a21, a22, TA, k); memcpy(TB, b11, k*k*sizeof(float)); } 
        else if (i == 2) { memcpy(TA, a11, k*k*sizeof(float)); sub(b12, b22, TB, k); } 
        else if (i == 3) { memcpy(TA, a22, k*k*sizeof(float)); sub(b21, b11, TB, k); } 
        else if (i == 4) { add(a11, a12, TA, k); memcpy(TB, b22, k*k*sizeof(float)); } 
        else if (i == 5) { sub(a21, a11, TA, k); add(b11, b12, TB, k); } 
        else if (i == 6) { sub(a12, a22, TA, k); add(b21, b22, TB, k); } 

        MPI_Bsend(&child_id, 1, MPI_LONG, target_rank, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(&k, 1, MPI_INT, target_rank, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(TA, k * k, MPI_FLOAT, target_rank, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(TB, k * k, MPI_FLOAT, target_rank, TAG_WORK, MPI_COMM_WORLD);
        
        delete[] TA; delete[] TB;
    }
    delete[] a11; delete[] a12; delete[] a21; delete[] a22;
    delete[] b11; delete[] b12; delete[] b21; delete[] b22;
}

void handle_result(long child_id, int k, float* M, int my_rank) {
    long parent_id = (child_id - 1) / 7;
    int index = (child_id - 1) % 7;
    if (ongoing_tasks.find(parent_id) == ongoing_tasks.end()) return;

    TaskState& state = ongoing_tasks[parent_id];
    state.results[index] = new float[k*k];
    memcpy(state.results[index], M, k * k * sizeof(float));
    state.count++;

    my_current_memory += (k * k);
    broadcast_memory_update(my_rank);

    if (state.count == 7) {
        assemble_and_send(parent_id, state, my_rank);
    }
}

void node_loop(int rank, int num_procs) {
    node_memory.resize(num_procs, 0);
    
    // Use SmartBuffer for persistent INPUTS
    SmartBuffer bufA, bufB;

    while (true) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;

        if (tag == TAG_TERMINATE) break;

        if (tag == TAG_WORK) {
            long mid; int n;
            MPI_Recv(&mid, 1, MPI_LONG, source, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&n, 1, MPI_INT, source, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            bufA.ensure_size(n);
            bufB.ensure_size(n);
            
            MPI_Recv(bufA.data, n * n, MPI_FLOAT, source, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(bufB.data, n * n, MPI_FLOAT, source, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            handle_new_work(source, mid, n, bufA.data, bufB.data, rank, num_procs);
        }
        else if (tag == TAG_RESULT) {
            long mid; int k;
            MPI_Recv(&mid, 1, MPI_LONG, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&k, 1, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            float* M = new float[k*k];
            MPI_Recv(M, k * k, MPI_FLOAT, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            handle_result(mid, k, M, rank);
            delete[] M;
        }
        else if (tag == TAG_MEM_UPDATE) {
            long payload[2];
            MPI_Recv(payload, 2, MPI_LONG, source, TAG_MEM_UPDATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if ((int)payload[0] < node_memory.size()) node_memory[(int)payload[0]] = payload[1];
            if (rank == 0) {
                for (int i = 1; i < num_procs; i++) {
                    if (i != source) MPI_Bsend(payload, 2, MPI_LONG, i, TAG_MEM_UPDATE, MPI_COMM_WORLD);
                }
            }
        }
    }
}

// ... Main function remains the same as previous safe version ...
// (Using 200MB buffer safety check)
int main(int argc, char** argv) {
    int rank, num_procs, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // 1. Buffer Setup
    size_t bsend_size = 200LL * 1024 * 1024; // 200MB Safe Buffer
    void* buffer = malloc(bsend_size);
    if (!buffer) { cout << "OOM"; return 1; }
    MPI_Buffer_attach(buffer, (int)bsend_size);

    srand(time(NULL) + rank);

    if (rank == 0) {
        int N;
        if(num_procs < 2) { cout << "Need 2+ procs"; return 1; }
        cout << "Enter N: "; cin >> N;
        
        // 2. Allocations
        float *A = new float[N*N];
        float *B = new float[N*N];
        float *FinalC = new float[N*N]; // [FIX] Allocate explicit result matrix

        // 3. Initialize Data
        #pragma omp parallel for
        for(int i=0; i<N*N; i++) { 
            A[i] = 1.0f; 
            B[i] = 1.0f; 
        } 

        // 4. Start Timer & Send
        cout << "Master: Start..." << endl;
        double start = MPI_Wtime();
        
        long root = 0; int target = 1;
        MPI_Bsend(&root, 1, MPI_LONG, target, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(&N, 1, MPI_INT, target, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(A, N*N, MPI_FLOAT, target, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(B, N*N, MPI_FLOAT, target, TAG_WORK, MPI_COMM_WORLD);
        
        node_memory.resize(num_procs, 0);
        bool done = false;
        double start_time = MPI_Wtime();
        // 5. Master Loop
        while(!done) {
             MPI_Status s; 
             MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &s);
             
             if (s.MPI_TAG == TAG_RESULT) {
                 long id; int k;
                 MPI_Recv(&id, 1, MPI_LONG, s.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                 MPI_Recv(&k, 1, MPI_INT, s.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                 
                 // [FIX] Receive directly into FinalC if it is the root result
                 if (id == 0) {
                     MPI_Recv(FinalC, N*N, MPI_FLOAT, s.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     cout << "Done in " << MPI_Wtime() - start << "s" << endl;
                     done = true;
                 } else {
                     // Flush other results (shouldn't happen for Master, but good for safety)
                     float* dummy = new float[k*k];
                     MPI_Recv(dummy, k*k, MPI_FLOAT, s.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     delete[] dummy;
                 }
             } 
             else if (s.MPI_TAG == TAG_MEM_UPDATE) {
                 long p[2]; 
                 MPI_Recv(p, 2, MPI_LONG, s.MPI_SOURCE, TAG_MEM_UPDATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                 for(int i=1; i<num_procs; i++) {
                     if(i != s.MPI_SOURCE) MPI_Bsend(p, 2, MPI_LONG, i, TAG_MEM_UPDATE, MPI_COMM_WORLD);
                 }
             }
        }
        double end = MPI_Wtime();
        cout << "Total Time: " << end - start << "s" << endl;
        // 6. Verify & Cleanup
        // verify_result(A, B, FinalC, N); // Make sure to uncomment your verification function helper
        
        // Simple manual verification check:
        // Since A=1, B=1, Result should be N everywhere.
        // if (FinalC[0] == (float)N) cout << "SUCCESS: C[0] == " << N << endl;
        // else cout << "FAIL: C[0] == " << FinalC[0] << " Expected " << N << endl;
        
        for(int i=1; i<num_procs; i++) MPI_Send(0,0,MPI_INT,i,TAG_TERMINATE,MPI_COMM_WORLD);
        
        delete[] A; delete[] B; delete[] FinalC;
    } else {
        node_loop(rank, num_procs);
    }

    MPI_Buffer_detach(&buffer, (int*)&bsend_size);
    free(buffer);
    MPI_Finalize();
    return 0;
}