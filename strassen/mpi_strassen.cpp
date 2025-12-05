#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring> // memset
#include <mpi.h>

using namespace std;

// --- CẤU HÌNH ---
const int CUTOFF = 128;

// --- PHẦN 1: QUẢN LÝ BỘ NHỚ ---

int* allocate_matrix(int n) {
    
    int* M = new int[n * n](); 
    for (int i = 0; i < n * n; i++) {
        M[i] = 5 - rand() % 10; // Giá trị ngẫu nhiên từ -4 đến 5
    }
    return M;
}

void free_matrix(int* M) {
    delete[] M;
}

// Reset ma trận về 0 để tái sử dụng
void reset_matrix(int* M, int n) {
    memset(M, 0, n * n * sizeof(int));
}

// --- PHẦN 2: CÁC PHÉP TOÁN CƠ BẢN ---

// Nhân thường tuần tự (Sequential) - Tối ưu cache i-k-j
void multiply_standard(const int* A, const int* B, int* C, int n, int stride) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            int temp = A[i * stride + k];
            for (int j = 0; j < n; j++) {
                C[i * stride + j] += temp * B[k * stride + j];
            }
        }
    }
}

// Nhân thường song song (Parallel Standard) - Tối ưu cache i-k-j + OpenMP
void run_parallel_standard(const int* A, const int* B, int* C, int n) {
    // Chỉ cần thêm pragma này để chia vòng lặp i cho các luồng
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            int temp = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += temp * B[k * n + j];
            }
        }
    }
}

void add(const int* A, const int* B, int* C, int n) {
    int size = n * n;
    for (int i = 0; i < size; i++) C[i] = A[i] + B[i];
}

void sub(const int* A, const int* B, int* C, int n) {
    int size = n * n;
    for (int i = 0; i < size; i++) C[i] = A[i] - B[i];
}

// --- PHẦN 3: GIẢI THUẬT STRASSEN (OPENMP) ---
// (Giữ nguyên logic của bạn, chỉ rút gọn để dễ nhìn)

void strassen_recursive(int* A, int* B, int* C, int n) {
    if (n <= CUTOFF) {
        multiply_standard(A, B, C, n, n);
        return;
    }

    int k = n / 2;
    
    int *a11 = allocate_matrix(k), *a12 = allocate_matrix(k), *a21 = allocate_matrix(k), *a22 = allocate_matrix(k);
    int *b11 = allocate_matrix(k), *b12 = allocate_matrix(k), *b21 = allocate_matrix(k), *b22 = allocate_matrix(k);
    int *p1 = allocate_matrix(k), *p2 = allocate_matrix(k), *p3 = allocate_matrix(k), *p4 = allocate_matrix(k);
    int *p5 = allocate_matrix(k), *p6 = allocate_matrix(k), *p7 = allocate_matrix(k);
    
    #pragma omp parallel for
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            a11[i * k + j] = A[i * n + j];          a12[i * k + j] = A[i * n + (j + k)];
            a21[i * k + j] = A[(i + k) * n + j];    a22[i * k + j] = A[(i + k) * n + (j + k)];
            b11[i * k + j] = B[i * n + j];          b12[i * k + j] = B[i * n + (j + k)];
            b21[i * k + j] = B[(i + k) * n + j];    b22[i * k + j] = B[(i + k) * n + (j + k)];
        }
    }
    
    #pragma omp task shared(a11, a22, b11, b22, p1)
    {
        int *tA = allocate_matrix(k), *tB = allocate_matrix(k);
        add(a11, a22, tA, k); add(b11, b22, tB, k);
        strassen_recursive(tA, tB, p1, k);
        free_matrix(tA); free_matrix(tB);
    }
    #pragma omp task shared(a21, a22, b11, p2)
    {
        int *tA = allocate_matrix(k);
        add(a21, a22, tA, k); strassen_recursive(tA, b11, p2, k); free_matrix(tA);
    }
    #pragma omp task shared(a11, b12, b22, p3)
    {
        int *tB = allocate_matrix(k);
        sub(b12, b22, tB, k); strassen_recursive(a11, tB, p3, k); free_matrix(tB);
    }
    #pragma omp task shared(a22, b21, b11, p4)
    {
        int *tB = allocate_matrix(k);
        sub(b21, b11, tB, k); strassen_recursive(a22, tB, p4, k); free_matrix(tB);
    }
    #pragma omp task shared(a11, a12, b22, p5)
    {
        int *tA = allocate_matrix(k);
        add(a11, a12, tA, k); strassen_recursive(tA, b22, p5, k); free_matrix(tA);
    }
    #pragma omp task shared(a21, a11, b11, b12, p6)
    {
        int *tA = allocate_matrix(k), *tB = allocate_matrix(k);
        sub(a21, a11, tA, k); add(b11, b12, tB, k);
        strassen_recursive(tA, tB, p6, k);
        free_matrix(tA); free_matrix(tB);
    }
    #pragma omp task shared(a12, a22, b21, b22, p7)
    {
        int *tA = allocate_matrix(k), *tB = allocate_matrix(k);
        sub(a12, a22, tA, k); add(b21, b22, tB, k);
        strassen_recursive(tA, tB, p7, k);
        free_matrix(tA); free_matrix(tB);
    }
    #pragma omp taskwait

    #pragma omp parallel for
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            C[i * n + j] = p1[i * k + j] + p4[i * k + j] - p5[i * k + j] + p7[i * k + j];
            C[i * n + (j + k)] = p3[i * k + j] + p5[i * k + j];
            C[(i + k) * n + j] = p2[i * k + j] + p4[i * k + j];
            C[(i + k) * n + (j + k)] = p1[i * k + j] - p2[i * k + j] + p3[i * k + j] + p6[i * k + j];
        }
    }

    free_matrix(a11); free_matrix(a12); free_matrix(a21); free_matrix(a22);
    free_matrix(b11); free_matrix(b12); free_matrix(b21); free_matrix(b22);
    free_matrix(p1); free_matrix(p2); free_matrix(p3); free_matrix(p4);
    free_matrix(p5); free_matrix(p6); free_matrix(p7);
}

void mpi_strassen(int* A, int* B, int* C, int n, int rank) {
    int k = n / 2;
    
    int *a11 = allocate_matrix(k), *a12 = allocate_matrix(k), *a21 = allocate_matrix(k), *a22 = allocate_matrix(k);
    int *b11 = allocate_matrix(k), *b12 = allocate_matrix(k), *b21 = allocate_matrix(k), *b22 = allocate_matrix(k);
    int *p1 = allocate_matrix(k), *p2 = allocate_matrix(k), *p3 = allocate_matrix(k), *p4 = allocate_matrix(k);
    int *p5 = allocate_matrix(k), *p6 = allocate_matrix(k), *p7 = allocate_matrix(k);
    
    #pragma omp parallel for
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            a11[i * k + j] = A[i * n + j];          a12[i * k + j] = A[i * n + (j + k)];
            a21[i * k + j] = A[(i + k) * n + j];    a22[i * k + j] = A[(i + k) * n + (j + k)];
            b11[i * k + j] = B[i * n + j];          b12[i * k + j] = B[i * n + (j + k)];
            b21[i * k + j] = B[(i + k) * n + j];    b22[i * k + j] = B[(i + k) * n + (j + k)];
        }
    }

    delete [] A;
    delete [] B;
    
    if(rank == 0){
        printf("Master process waiting for results...\n");
        MPI_Recv(p1, k*k, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p2, k*k, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p3, k*k, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p4, k*k, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p5, k*k, MPI_INT, 5, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p6, k*k, MPI_INT, 6, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p7, k*k, MPI_INT, 7, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if(rank == 1){
        int *tA = allocate_matrix(k), *tB = allocate_matrix(k);
        add(a11, a22, tA, k); add(b11, b22, tB, k);
        strassen_recursive(tA, tB, p1, k);
        free_matrix(tA); free_matrix(tB);
        MPI_Send(p1, k*k, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if(rank == 2){
        int *tA = allocate_matrix(k);
        add(a21, a22, tA, k); strassen_recursive(tA, b11, p2, k); free_matrix(tA);
        MPI_Send(p2, k*k, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if(rank == 3){
        int *tB = allocate_matrix(k);
        sub(b12, b22, tB, k); strassen_recursive(a11, tB, p3, k); free_matrix(tB);
        MPI_Send(p3, k*k, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if(rank == 4){
        int *tB = allocate_matrix(k);
        sub(b21, b11, tB, k); strassen_recursive(a22, tB, p4, k); free_matrix(tB);
        MPI_Send(p4, k*k, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if(rank == 5){
        int *tA = allocate_matrix(k);
        add(a11, a12, tA, k); strassen_recursive(tA, b22, p5, k); free_matrix(tA);
        MPI_Send(p5, k*k, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if(rank == 6){
        int *tA = allocate_matrix(k), *tB = allocate_matrix(k);
        sub(a21, a11, tA, k); add(b11, b12, tB, k);
        strassen_recursive(tA, tB, p6, k);
        free_matrix(tA); free_matrix(tB);
        MPI_Send(p6, k*k, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if(rank == 7){
        int *tA = allocate_matrix(k), *tB = allocate_matrix(k);
        sub(a12, a22, tA, k); add(b21, b22, tB, k);
        strassen_recursive(tA, tB, p7, k);
        free_matrix(tA); free_matrix(tB);
        MPI_Send(p7, k*k, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
        #pragma omp parallel for
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                C[i * n + j] = p1[i * k + j] + p4[i * k + j] - p5[i * k + j] + p7[i * k + j];
                C[i * n + (j + k)] = p3[i * k + j] + p5[i * k + j];
                C[(i + k) * n + j] = p2[i * k + j] + p4[i * k + j];
                C[(i + k) * n + (j + k)] = p1[i * k + j] - p2[i * k + j] + p3[i * k + j] + p6[i * k + j];
            }
        }
    }

    free_matrix(a11); free_matrix(a12); free_matrix(a21); free_matrix(a22);
    free_matrix(b11); free_matrix(b12); free_matrix(b21); free_matrix(b22);
    free_matrix(p1); free_matrix(p2); free_matrix(p3); free_matrix(p4);
    free_matrix(p5); free_matrix(p6); free_matrix(p7);

}

// --- PHẦN 4: GIẢI THUẬT TUẦN TỰ (SEQUENTIAL) ---

void run_sequential_multiplication(const int* A, const int* B, int* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            int temp = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += temp * B[k * n + j];
            }
        }
    }
}


bool check_result(const int* C_Test, const int* C_Ref, int real_n, int pad_n) {
    for (int i = 0; i < real_n; i++) {
        for (int j = 0; j < real_n; j++) {
            if (C_Test[i * pad_n + j] != C_Ref[i * pad_n + j]) return false;
        }
    }
    return true;
}


// --- MAIN ---

int main(int argc, char** argv) {
    int p_rank;
    int num_process;

    if(MPI_Init(&argc, &argv) != MPI_SUCCESS){
        cout << "MPI Initialization failed!" << endl;
        return -1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);

    int n;
    if(p_rank == 0){
        cout << "Enter matrix size (n x n): ";
        cin >> n;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int* A = allocate_matrix(n);
    int* B = allocate_matrix(n);

    MPI_Bcast(A, n*n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, n*n, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    int *C_MpiStrassen = nullptr;
    if(p_rank == 0){
        C_MpiStrassen = allocate_matrix(n);
    }
    mpi_strassen(A, B, C_MpiStrassen, n, p_rank);
    double end_time = MPI_Wtime();
    if (p_rank == 0) {
        cout << "MPI Strassen Time: " << end_time - start_time << " seconds" << endl;
    }
    MPI_Finalize();
    return 0;
}