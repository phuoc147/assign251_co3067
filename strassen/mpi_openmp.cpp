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
const int MPI_RECURSION_LIMIT = 512; // Must be < N to trigger distribution
const int TAG_WORK = 0;
const int TAG_RESULT = 1;
const int TAG_MEM_UPDATE = 4;
const int TAG_TERMINATE = 99;

// --- GLOBAL STATE ---
struct TaskState {
    int count;             
    int* results[7];       
    int n;                 
    int parent_rank;       

    TaskState() {
        count = 0;
        for (int i = 0; i < 7; i++) results[i] = nullptr;
        n = 0;
        parent_rank = -1;
    }
};

std::map<long, TaskState> ongoing_tasks;
std::vector<long> node_memory; 
long my_current_memory = 0;    

// --- HELPERS ---
void log_debug(int rank, string msg) {
    // [ENABLED] Printing to stdout
    printf("[DEBUG] Rank %d: %s\n", rank, msg.c_str());
    fflush(stdout); // Force print immediately
}

int* allocate_empty(int n) { return new int[n * n](); }

int* allocate_random(int n) {
    int* M = new int[n * n];
    for (int i = 0; i < n * n; i++) M[i] = rand() % 5;
    return M;
}

void free_m(int* m) { if (m) delete[] m; }

// --- MATH KERNELS ---
void add(const int* A, const int* B, int* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) C[i] = A[i] + B[i];
}

void sub(const int* A, const int* B, int* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) C[i] = A[i] - B[i];
}

void multiply_std(const int* A, const int* B, int* C, int n) {
    memset(C, 0, n * n * sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            int temp = A[i * n + k];
            for (int j = 0; j < n; j++) C[i * n + j] += temp * B[k * n + j];
        }
    }
}

// --- LOCAL STRASSEN ---
void local_strassen_recursive(int* A, int* B, int* C, int n) {
    if (n <= CUTOFF) {
        multiply_std(A, B, C, n);
        return;
    }
    int k = n / 2;
    int *a11 = allocate_empty(k), *a12 = allocate_empty(k), *a21 = allocate_empty(k), *a22 = allocate_empty(k);
    int *b11 = allocate_empty(k), *b12 = allocate_empty(k), *b21 = allocate_empty(k), *b22 = allocate_empty(k);
    int *p1 = allocate_empty(k), *p2 = allocate_empty(k), *p3 = allocate_empty(k), *p4 = allocate_empty(k);
    int *p5 = allocate_empty(k), *p6 = allocate_empty(k), *p7 = allocate_empty(k);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            a11[i*k+j] = A[i*n+j];       a12[i*k+j] = A[i*n+(j+k)];
            a21[i*k+j] = A[(i+k)*n+j];   a22[i*k+j] = A[(i+k)*n+(j+k)];
            b11[i*k+j] = B[i*n+j];       b12[i*k+j] = B[i*n+(j+k)];
            b21[i*k+j] = B[(i+k)*n+j];   b22[i*k+j] = B[(i+k)*n+(j+k)];
        }
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            { int *tA = allocate_empty(k), *tB = allocate_empty(k); add(a11, a22, tA, k); add(b11, b22, tB, k); local_strassen_recursive(tA, tB, p1, k); free_m(tA); free_m(tB); }
            #pragma omp task
            { int *tA = allocate_empty(k); add(a21, a22, tA, k); local_strassen_recursive(tA, b11, p2, k); free_m(tA); }
            #pragma omp task
            { int *tB = allocate_empty(k); sub(b12, b22, tB, k); local_strassen_recursive(a11, tB, p3, k); free_m(tB); }
            #pragma omp task
            { int *tB = allocate_empty(k); sub(b21, b11, tB, k); local_strassen_recursive(a22, tB, p4, k); free_m(tB); }
            #pragma omp task
            { int *tA = allocate_empty(k); add(a11, a12, tA, k); local_strassen_recursive(tA, b22, p5, k); free_m(tA); }
            #pragma omp task
            { int *tA = allocate_empty(k), *tB = allocate_empty(k); sub(a21, a11, tA, k); add(b11, b12, tB, k); local_strassen_recursive(tA, tB, p6, k); free_m(tA); free_m(tB); }
            #pragma omp task
            { int *tA = allocate_empty(k), *tB = allocate_empty(k); sub(a12, a22, tA, k); add(b21, b22, tB, k); local_strassen_recursive(tA, tB, p7, k); free_m(tA); free_m(tB); }
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            C[i*n+j]         = p1[i*k+j] + p4[i*k+j] - p5[i*k+j] + p7[i*k+j];
            C[i*n+(j+k)]     = p3[i*k+j] + p5[i*k+j];
            C[(i+k)*n+j]     = p2[i*k+j] + p4[i*k+j];
            C[(i+k)*n+(j+k)] = p1[i*k+j] - p2[i*k+j] + p3[i*k+j] + p6[i*k+j];
        }
    }
    free_m(a11); free_m(a12); free_m(a21); free_m(a22);
    free_m(b11); free_m(b12); free_m(b21); free_m(b22);
    free_m(p1); free_m(p2); free_m(p3); free_m(p4); free_m(p5); free_m(p6); free_m(p7);
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
        if (i != my_rank) {
            sorted_nodes.push_back({node_memory[i], i});
        }
    }
    
    // Shuffle equal values to prevent Thundering Herd on Rank 2
    std::random_shuffle(sorted_nodes.begin(), sorted_nodes.end());
    std::stable_sort(sorted_nodes.begin(), sorted_nodes.end(), 
        [](const std::pair<long, int>& a, const std::pair<long, int>& b) {
            return a.first < b.first;
    });

    std::vector<int> targets;
    if (sorted_nodes.empty()) return targets;

    int needed = 7;
    int available = sorted_nodes.size();
    
    for (int i = 0; i < std::min(needed, available); i++) {
        targets.push_back(sorted_nodes[i].second);
    }
    
    if (targets.size() > 0) {
        while (targets.size() < 7) {
            targets.push_back(targets[targets.size() % available]);
        }
    }
    return targets;
}

// --- HANDLERS ---
void assemble_and_send(long matrix_id, TaskState& state, int my_rank) {
    // log_debug(my_rank, "Assembling Result for ID " + to_string(matrix_id));
    int n = state.n;
    int k = n / 2;
    int* C = allocate_empty(n);
    int* p[7];
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
    MPI_Bsend(C, n * n, MPI_INT, state.parent_rank, TAG_RESULT, MPI_COMM_WORLD);

    free_m(C);
    for (int i = 0; i < 7; i++) free_m(state.results[i]);
    ongoing_tasks.erase(matrix_id);

    long freed = (7 * k * k) + (n * n);
    if (freed > my_current_memory) my_current_memory = 0; else my_current_memory -= freed;
    broadcast_memory_update(my_rank);
}

void handle_new_work(int source, long matrix_id, int n, int* A, int* B, int my_rank, int num_procs) {
    // log_debug(my_rank, "Received Work ID " + to_string(matrix_id) + " Size " + to_string(n));
    my_current_memory += (2L * n * n);
    broadcast_memory_update(my_rank);

    if (n <= CUTOFF) {
        // log_debug(my_rank, "Computing Base Case ID " + to_string(matrix_id));
        int* C = allocate_empty(n);
        multiply_std(A, B, C, n);
        
        MPI_Bsend(&matrix_id, 1, MPI_LONG, source, TAG_RESULT, MPI_COMM_WORLD);
        MPI_Bsend(&n, 1, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD);
        MPI_Bsend(C, n * n, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD);
        
        free_m(C);
        my_current_memory -= (2L * n * n);
        broadcast_memory_update(my_rank);
        return;
    }

    std::vector<int> targets = find_best_workers(num_procs, my_rank);
    
    // Switch to Local if limit reached or no peers
    if (targets.size() < 7 || n < MPI_RECURSION_LIMIT) {
        // log_debug(my_rank, "Switching to LOCAL Recursion for ID " + to_string(matrix_id));
        int* C = allocate_empty(n);
        local_strassen_recursive(A, B, C, n); 
        
        MPI_Bsend(&matrix_id, 1, MPI_LONG, source, TAG_RESULT, MPI_COMM_WORLD);
        MPI_Bsend(&n, 1, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD);
        MPI_Bsend(C, n * n, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD);
        
        free_m(C);
        my_current_memory -= (2L * n * n);
        broadcast_memory_update(my_rank);
        return;
    }

    // log_debug(my_rank, "Distributing ID " + to_string(matrix_id) + " to peers.");
    TaskState state;
    state.n = n;
    state.parent_rank = source;
    ongoing_tasks[matrix_id] = state;

    int k = n / 2;
    int *a11 = allocate_empty(k), *a12 = allocate_empty(k), *a21 = allocate_empty(k), *a22 = allocate_empty(k);
    int *b11 = allocate_empty(k), *b12 = allocate_empty(k), *b21 = allocate_empty(k), *b22 = allocate_empty(k);

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
        int *TA = allocate_empty(k), *TB = allocate_empty(k);

        if (i == 0) { add(a11, a22, TA, k); add(b11, b22, TB, k); }      
        else if (i == 1) { add(a21, a22, TA, k); memcpy(TB, b11, k*k*sizeof(int)); } 
        else if (i == 2) { memcpy(TA, a11, k*k*sizeof(int)); sub(b12, b22, TB, k); } 
        else if (i == 3) { memcpy(TA, a22, k*k*sizeof(int)); sub(b21, b11, TB, k); } 
        else if (i == 4) { add(a11, a12, TA, k); memcpy(TB, b22, k*k*sizeof(int)); } 
        else if (i == 5) { sub(a21, a11, TA, k); add(b11, b12, TB, k); } 
        else if (i == 6) { sub(a12, a22, TA, k); add(b21, b22, TB, k); } 

        MPI_Bsend(&child_id, 1, MPI_LONG, target_rank, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(&k, 1, MPI_INT, target_rank, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(TA, k * k, MPI_INT, target_rank, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(TB, k * k, MPI_INT, target_rank, TAG_WORK, MPI_COMM_WORLD);
        free_m(TA); free_m(TB);
    }
    free_m(a11); free_m(a12); free_m(a21); free_m(a22);
    free_m(b11); free_m(b12); free_m(b21); free_m(b22);
}

void handle_result(long child_id, int k, int* M, int my_rank) {
    long parent_id = (child_id - 1) / 7;
    int index = (child_id - 1) % 7;
    if (ongoing_tasks.find(parent_id) == ongoing_tasks.end()) return;

    TaskState& state = ongoing_tasks[parent_id];
    state.results[index] = allocate_empty(k);
    memcpy(state.results[index], M, k * k * sizeof(int));
    state.count++;

    my_current_memory += (k * k);
    broadcast_memory_update(my_rank);

    // log_debug(my_rank, "Received Part " + to_string(index+1) + "/7 for ID " + to_string(parent_id));

    if (state.count == 7) {
        assemble_and_send(parent_id, state, my_rank);
    }
}

void node_loop(int rank, int num_procs) {
    node_memory.resize(num_procs, 0);
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
            int* A = allocate_empty(n); int* B = allocate_empty(n);
            MPI_Recv(A, n * n, MPI_INT, source, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(B, n * n, MPI_INT, source, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            handle_new_work(source, mid, n, A, B, rank, num_procs);
            free_m(A); free_m(B);
        }
        else if (tag == TAG_RESULT) {
            long mid; int k;
            MPI_Recv(&mid, 1, MPI_LONG, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&k, 1, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int* M = allocate_empty(k);
            MPI_Recv(M, k * k, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            handle_result(mid, k, M, rank);
            free_m(M);
        }
        else if (tag == TAG_MEM_UPDATE) {
            long payload[2];
            MPI_Recv(payload, 2, MPI_LONG, source, TAG_MEM_UPDATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if ((int)payload[0] < node_memory.size()) node_memory[(int)payload[0]] = payload[1];
            if (rank == 0) {
                for (int i = 1; i < num_procs; i++) {
                    if (i != source) {
                        MPI_Bsend(payload, 2, MPI_LONG, i, TAG_MEM_UPDATE, MPI_COMM_WORLD);
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int rank, num_procs, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int buf_size = 512 * 1024 * 1024; 
    void* buffer = malloc(buf_size);
    MPI_Buffer_attach(buffer, buf_size);

    srand(time(NULL) + rank);

    if (num_procs < 2) {
        if (rank == 0) cout << "Error: Run with -np 2 or more." << endl;
        MPI_Buffer_detach(&buffer, &buf_size);
        free(buffer);
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        int N;
        cout << "Enter N (e.g. 1024): ";
        cin >> N;
        int* A = allocate_random(N);
        int* B = allocate_random(N);
        
        cout << "Master: Sending Start to Worker 1..." << endl;
        double start = MPI_Wtime();

        long root_id = 0;
        int worker_target = 1; 
        
        MPI_Bsend(&root_id, 1, MPI_LONG, worker_target, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(&N, 1, MPI_INT, worker_target, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(A, N * N, MPI_INT, worker_target, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(B, N * N, MPI_INT, worker_target, TAG_WORK, MPI_COMM_WORLD);

        node_memory.resize(num_procs, 0);
        int* FinalC = allocate_empty(N);
        bool done = false;

        while (!done) {
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int source = status.MPI_SOURCE;
            int tag = status.MPI_TAG;

            if (tag == TAG_RESULT) {
                long mid; int k;
                MPI_Recv(&mid, 1, MPI_LONG, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&k, 1, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                if (mid == 0 && k == N) {
                    MPI_Recv(FinalC, N * N, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cout << "Master: Received Final Result!" << endl;
                    done = true;
                } else {
                    int* dummy = allocate_empty(k);
                    MPI_Recv(dummy, k*k, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    free_m(dummy);
                }
            } else if (tag == TAG_MEM_UPDATE) {
                long payload[2];
                MPI_Recv(payload, 2, MPI_LONG, source, TAG_MEM_UPDATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if ((int)payload[0] < node_memory.size()) node_memory[(int)payload[0]] = payload[1];
                for (int i = 1; i < num_procs; i++) {
                    if (i != source) {
                        MPI_Bsend(payload, 2, MPI_LONG, i, TAG_MEM_UPDATE, MPI_COMM_WORLD);
                    }
                }
            }
        }

        double end = MPI_Wtime();
        cout << "Time: " << end - start << "s" << endl;
        for (int i = 1; i < num_procs; i++) MPI_Send(NULL, 0, MPI_INT, i, TAG_TERMINATE, MPI_COMM_WORLD);
        
        free_m(A); free_m(B); free_m(FinalC);
    } 
    else {
        node_loop(rank, num_procs);
    }

    MPI_Buffer_detach(&buffer, &buf_size);
    free(buffer);
    MPI_Finalize();
    return 0;
}