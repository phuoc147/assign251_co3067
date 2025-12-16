#include <iostream>
#include <vector>
#include <map>
#include <mpi.h>
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <cmath>

using namespace std;

const int CUTOFF = 64; 
const int MPI_RECURSION_LIMIT = 2048; 
const int TAG_WORK = 0;
const int TAG_RESULT = 1;
const int TAG_MEM_UPDATE = 4;
const int TAG_TERMINATE = 99;

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

void verify_result(float* A, float* B, float* C, int n) {
    cout << "--- Verification (Random Check) ---" << endl;
    int r = rand() % n;
    int c = rand() % n;
    float expected = 0;
    for(int i=0; i<n; i++) expected += A[r*n + i] * B[i*n + c];
    
    float actual = C[r*n + c];
    if (abs(expected - actual) < 1e-2) 
        cout << "PASS: [" << r << "," << c << "] Expected: " << expected << " Got: " << actual << endl;
    else 
        cout << "FAIL: [" << r << "," << c << "] Expected: " << expected << " Got: " << actual << endl;
}

void add(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) C[i] = A[i] + B[i];
}

void sub(const float* A, const float* B, float* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) C[i] = A[i] - B[i];
}

void multiply_std(const float* A, const float* B, float* C, int n) {
    memset(C, 0, n * n * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float temp = A[i * n + k];
            for (int j = 0; j < n; j++) C[i * n + j] += temp * B[k * n + j];
        }
    }
}

void local_strassen_recursive(float* A, float* B, float* C, int n) {
    if (n <= CUTOFF) {
        multiply_std(A, B, C, n);
        return;
    }
    int k = n / 2;
    float *a[2][2], *b[2][2], *p[8];
    for(int i=0; i<2; i++) for(int j=0; j<2; j++) { a[i][j] = new float[k*k]; b[i][j] = new float[k*k]; }
    for(int i=1; i<=7; i++) p[i] = new float[k*k];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            a[0][0][i*k+j] = A[i*n+j];       a[0][1][i*k+j] = A[i*n+(j+k)];
            a[1][0][i*k+j] = A[(i+k)*n+j];   a[1][1][i*k+j] = A[(i+k)*n+(j+k)];
            b[0][0][i*k+j] = B[i*n+j];       b[0][1][i*k+j] = B[i*n+(j+k)];
            b[1][0][i*k+j] = B[(i+k)*n+j];   b[1][1][i*k+j] = B[(i+k)*n+(j+k)];
        }
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            { float *tA = new float[k*k], *tB = new float[k*k]; add(a[0][0], a[1][1], tA, k); add(b[0][0], b[1][1], tB, k); local_strassen_recursive(tA, tB, p[1], k); delete[] tA; delete[] tB; }
            #pragma omp task
            { float *tA = new float[k*k]; add(a[1][0], a[1][1], tA, k); local_strassen_recursive(tA, b[0][0], p[2], k); delete[] tA; }
            #pragma omp task
            { float *tB = new float[k*k]; sub(b[0][1], b[1][1], tB, k); local_strassen_recursive(a[0][0], tB, p[3], k); delete[] tB; }
            #pragma omp task
            { float *tB = new float[k*k]; sub(b[1][0], b[0][0], tB, k); local_strassen_recursive(a[1][1], tB, p[4], k); delete[] tB; }
            #pragma omp task
            { float *tA = new float[k*k]; add(a[0][0], a[0][1], tA, k); local_strassen_recursive(tA, b[1][1], p[5], k); delete[] tA; }
            #pragma omp task
            { float *tA = new float[k*k], *tB = new float[k*k]; sub(a[1][0], a[0][0], tA, k); add(b[0][0], b[0][1], tB, k); local_strassen_recursive(tA, tB, p[6], k); delete[] tA; delete[] tB; }
            #pragma omp task
            { float *tA = new float[k*k], *tB = new float[k*k]; sub(a[0][1], a[1][1], tA, k); add(b[1][0], b[1][1], tB, k); local_strassen_recursive(tA, tB, p[7], k); delete[] tA; delete[] tB; }
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            C[i*n+j]         = p[1][i*k+j] + p[4][i*k+j] - p[5][i*k+j] + p[7][i*k+j];
            C[i*n+(j+k)]     = p[3][i*k+j] + p[5][i*k+j];
            C[(i+k)*n+j]     = p[2][i*k+j] + p[4][i*k+j];
            C[(i+k)*n+(j+k)] = p[1][i*k+j] - p[2][i*k+j] + p[3][i*k+j] + p[6][i*k+j];
        }
    }
    for(int i=0; i<2; i++) for(int j=0; j<2; j++) { delete[] a[i][j]; delete[] b[i][j]; }
    for(int i=1; i<=7; i++) delete[] p[i];
}

void broadcast_memory_update(int my_rank) {
    if (my_rank < (int)node_memory.size()) node_memory[my_rank] = my_current_memory;
    long payload[2] = {(long)my_rank, my_current_memory};
    if (my_rank > 0) MPI_Bsend(payload, 2, MPI_LONG, 0, TAG_MEM_UPDATE, MPI_COMM_WORLD);
}

std::vector<int> find_best_workers(int num_procs, int my_rank) {
    std::vector<std::pair<long, int>> sorted_nodes;
    for (int i = 1; i < num_procs; i++) if (i != my_rank) sorted_nodes.push_back({node_memory[i], i});
    random_shuffle(sorted_nodes.begin(), sorted_nodes.end());
    stable_sort(sorted_nodes.begin(), sorted_nodes.end(), [](const pair<long, int>& a, const pair<long, int>& b) { return a.first < b.first; });
    std::vector<int> targets;
    if (sorted_nodes.empty()) return targets;
    while (targets.size() < 7) targets.push_back(sorted_nodes[targets.size() % sorted_nodes.size()].second);
    return targets;
}

void assemble_and_send(long matrix_id, TaskState& state, int my_rank) {
    int n = state.n; int k = n / 2;
    float* C = new float[n*n]; 
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            C[i*n+j]         = state.results[0][i*k+j] + state.results[3][i*k+j] - state.results[4][i*k+j] + state.results[6][i*k+j];
            C[i*n+(j+k)]     = state.results[2][i*k+j] + state.results[4][i*k+j];
            C[(i+k)*n+j]     = state.results[1][i*k+j] + state.results[3][i*k+j];
            C[(i+k)*n+(j+k)] = state.results[0][i*k+j] - state.results[1][i*k+j] + state.results[2][i*k+j] + state.results[5][i*k+j];
        }
    }
    MPI_Bsend(&matrix_id, 1, MPI_LONG, state.parent_rank, TAG_RESULT, MPI_COMM_WORLD);
    MPI_Bsend(&n, 1, MPI_INT, state.parent_rank, TAG_RESULT, MPI_COMM_WORLD);
    MPI_Bsend(C, n * n, MPI_FLOAT, state.parent_rank, TAG_RESULT, MPI_COMM_WORLD);
    delete[] C;
    for (int i = 0; i < 7; i++) delete[] state.results[i];
    ongoing_tasks.erase(matrix_id);
    my_current_memory -= ((7L * k * k) + (n * n));
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
    TaskState state; state.n = n; state.parent_rank = source;
    ongoing_tasks[matrix_id] = state;
    int k = n / 2;
    float *a[2][2], *b[2][2];
    for(int i=0; i<2; i++) for(int j=0; j<2; j++) { a[i][j] = new float[k*k]; b[i][j] = new float[k*k]; }
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            a[0][0][i*k+j] = A[i*n+j];       a[0][1][i*k+j] = A[i*n+(j+k)];
            a[1][0][i*k+j] = A[(i+k)*n+j];   a[1][1][i*k+j] = A[(i+k)*n+(j+k)];
            b[0][0][i*k+j] = B[i*n+j];       b[0][1][i*k+j] = B[i*n+(j+k)];
            b[1][0][i*k+j] = B[(i+k)*n+j];   b[1][1][i*k+j] = B[(i+k)*n+(j+k)];
        }
    }
    for (int i = 0; i < 7; i++) {
        long child_id = (matrix_id * 7) + (i + 1);
        float *TA = new float[k*k], *TB = new float[k*k];
        if (i == 0) { add(a[0][0], a[1][1], TA, k); add(b[0][0], b[1][1], TB, k); }      
        else if (i == 1) { add(a[1][0], a[1][1], TA, k); memcpy(TB, b[0][0], k*k*sizeof(float)); } 
        else if (i == 2) { memcpy(TA, a[0][0], k*k*sizeof(float)); sub(b[0][1], b[1][1], TB, k); } 
        else if (i == 3) { memcpy(TA, a[1][1], k*k*sizeof(float)); sub(b[1][0], b[0][0], TB, k); } 
        else if (i == 4) { add(a[0][0], a[0][1], TA, k); memcpy(TB, b[1][1], k*k*sizeof(float)); } 
        else if (i == 5) { sub(a[1][0], a[0][0], TA, k); add(b[0][0], b[0][1], TB, k); } 
        else if (i == 6) { sub(a[0][1], a[1][1], TA, k); add(b[1][0], b[1][1], TB, k); } 
        MPI_Bsend(&child_id, 1, MPI_LONG, targets[i], TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(&k, 1, MPI_INT, targets[i], TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(TA, k * k, MPI_FLOAT, targets[i], TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(TB, k * k, MPI_FLOAT, targets[i], TAG_WORK, MPI_COMM_WORLD);
        delete[] TA; delete[] TB;
    }
    for(int i=0; i<2; i++) for(int j=0; j<2; j++) { delete[] a[i][j]; delete[] b[i][j]; }
}

void node_loop(int rank, int num_procs) {
    node_memory.resize(num_procs, 0);
    SmartBuffer bufA, bufB;
    while (true) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TAG_TERMINATE) break;
        if (status.MPI_TAG == TAG_WORK) {
            long mid; int n;
            MPI_Recv(&mid, 1, MPI_LONG, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&n, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            bufA.ensure_size(n); bufB.ensure_size(n);
            MPI_Recv(bufA.data, n * n, MPI_FLOAT, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(bufB.data, n * n, MPI_FLOAT, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            handle_new_work(status.MPI_SOURCE, mid, n, bufA.data, bufB.data, rank, num_procs);
        } else if (status.MPI_TAG == TAG_RESULT) {
            long mid; int k;
            MPI_Recv(&mid, 1, MPI_LONG, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&k, 1, MPI_INT, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            float* M = new float[k*k];
            MPI_Recv(M, k * k, MPI_FLOAT, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            long parent_id = (mid - 1) / 7;
            if (ongoing_tasks.count(parent_id)) {
                ongoing_tasks[parent_id].results[(mid - 1) % 7] = M;
                ongoing_tasks[parent_id].count++;
                my_current_memory += (k * k);
                broadcast_memory_update(rank);
                if (ongoing_tasks[parent_id].count == 7) assemble_and_send(parent_id, ongoing_tasks[parent_id], rank);
            } else delete[] M;
        } else if (status.MPI_TAG == TAG_MEM_UPDATE) {
            long p[2]; MPI_Recv(p, 2, MPI_LONG, status.MPI_SOURCE, TAG_MEM_UPDATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (p[0] < (int)node_memory.size()) node_memory[p[0]] = p[1];
            if (rank == 0) for (int i = 1; i < num_procs; i++) if (i != status.MPI_SOURCE) MPI_Bsend(p, 2, MPI_LONG, i, TAG_MEM_UPDATE, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char** argv) {
    int rank, num_procs, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    int N = 0;
    if (rank == 0) { cout << "Enter N: "; cin >> N; }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    long long bsend_size = (16LL * (N/2) * (N/2) * sizeof(float)) + (1024LL * 1024 * 100); 
    if (bsend_size < 200LL * 1024 * 1024) bsend_size = 200LL * 1024 * 1024;
    void* buffer = malloc(bsend_size);
    MPI_Buffer_attach(buffer, (int)bsend_size);
    srand(time(NULL) + rank);
    if (rank == 0) {
        float *A = new float[N*N], *B = new float[N*N], *FinalC = new float[N*N];
        #pragma omp parallel for
        for(int i=0; i<N*N; i++) { A[i] = 1.0f; B[i] = 1.0f; } 
        double start = MPI_Wtime();
        long root = 0; MPI_Bsend(&root, 1, MPI_LONG, 1, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(&N, 1, MPI_INT, 1, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(A, N*N, MPI_FLOAT, 1, TAG_WORK, MPI_COMM_WORLD);
        MPI_Bsend(B, N*N, MPI_FLOAT, 1, TAG_WORK, MPI_COMM_WORLD);
        node_memory.resize(num_procs, 0); bool done = false;
        while(!done) {
            MPI_Status s; MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &s);
            if (s.MPI_TAG == TAG_RESULT) {
                long id; int k;
                MPI_Recv(&id, 1, MPI_LONG, s.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&k, 1, MPI_INT, s.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (id == 0) { MPI_Recv(FinalC, N*N, MPI_FLOAT, s.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE); done = true; }
                else { float* d = new float[k*k]; MPI_Recv(d, k*k, MPI_FLOAT, s.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE); delete[] d; }
            } else if (s.MPI_TAG == TAG_MEM_UPDATE) {
                long p[2]; MPI_Recv(p, 2, MPI_LONG, s.MPI_SOURCE, TAG_MEM_UPDATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(p[0] < (int)node_memory.size()) node_memory[p[0]] = p[1];
                for(int i=1; i<num_procs; i++) if(i != s.MPI_SOURCE) MPI_Bsend(p, 2, MPI_LONG, i, TAG_MEM_UPDATE, MPI_COMM_WORLD);
            }
        }
        cout << "Done in " << MPI_Wtime() - start << "s" << endl;
        // verify_result(A, B, FinalC, N);
        for(int i=1; i<num_procs; i++) MPI_Send(0,0,MPI_INT,i,TAG_TERMINATE,MPI_COMM_WORLD);
        delete[] A; delete[] B; delete[] FinalC;
    } else node_loop(rank, num_procs);
    MPI_Buffer_detach(&buffer, (int*)&bsend_size); free(buffer); MPI_Finalize(); return 0;
}