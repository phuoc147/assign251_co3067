#include <iostream>
#include <ctime>
#include <fstream>
#include <sys/time.h>
#include <mpi.h>

#define IDX(i, j, cols) ((i) * (cols) + (j))

bool saveMatrixToCSV(int* matrix, int rows, int cols, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return false;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outFile << matrix[IDX(i, j, cols)];

            if (j < cols - 1) {
                outFile << ",";
            }
        }
        outFile << "\n";
    }

    outFile.close();
    // std::cout << "Successfully saved matrix to " << filename << std::endl;
    return true;
}

void printMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[IDX(i, j, cols)] << " ";
        }
        std::cout << std::endl;
    }
}

int* createRandomMatrix(int rows, int cols) {
    // Seed the random number generator only once per program execution
    static bool seeded = false;
    if (!seeded) {
        std::srand(static_cast<unsigned int>(std::time(0)));
        seeded = true;
    }

    if (rows <= 0 || cols <= 0) {
        return nullptr;
    }

    // Allocate memory for rows * cols elements
    int* matrix = new int[rows * cols];

    // Initialize them
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = std::rand() % 10;
    }

    return matrix;
}

int* transposeMatrix(int* matrix, int size) {
    int* matrix_T = new int[size * size];

    // Transpose is safe to parallelize for large matrices
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // matrix_T[i][j] = matrix[j][i]
            matrix_T[IDX(i, j, size)] = matrix[IDX(j, i, size)];
        }
    }
    return matrix_T;
}

int* multiplication(int* matrix_A, int* matrix_B_T, int rows, int cols) {
    int* matrix_C_part = new int[rows * cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            
            int sum = 0;
            for (int k = 0; k < cols; k++) {
                sum += matrix_A[IDX(i, k, cols)] * matrix_B_T[IDX(j, k, cols)];
            }
            matrix_C_part[IDX(i, j, cols)] = sum;
        }
    }
    return matrix_C_part;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int N_size = 0;

    // --- Argument Validation (Only Rank 0 reads arguments) ---
    if (rank == 0) {
        
        // Process count validation
        if (num_procs < 4) {
            std::cerr << "Error: Invalid Process Count. This program expects the minimum processes is 4." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // argument count
        if (argc != 2) {
            std::cerr << "Error: Invalid Argument Count. This program expects one size parameter." << std::endl;
            std::cerr << "Usage: " << argv[0] << " <N_size>" << std::endl;
            std::cerr << "Example: " << argv[0] << " 10000" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        try {
            N_size = std::stoi(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid input for matrix size. Please enter an integer." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (N_size <= 0) {
            std::cerr << "Error: Matrix size must be a positive integer." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (N_size % num_procs != 0) {
            std::cerr << "Error: Invalid Argument Count. This program expects the process number must be dividable by the matrix size." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::cout << "MPI Matrix Multiplication (N=" << N_size << ") with " << num_procs << " processes." << std::endl;
    }

    // syncronize
    MPI_Barrier(MPI_COMM_WORLD);

    // -- 1. distribute N_size value to other processes
    MPI_Bcast(&N_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // std::cout << "Process " << rank << " receive matrix size: " << N_size << std::endl;

    // syncronize
    MPI_Barrier(MPI_COMM_WORLD);

    int* matrix_A = nullptr;
    int* matrix_B = nullptr;
    int* matrix_B_T = nullptr;
    int* matrix_C = nullptr;
    
    // -- 2. initialize data for 2 matrix
    if (rank == 0) {
        matrix_A = createRandomMatrix(N_size, N_size);
        saveMatrixToCSV(matrix_A, N_size, N_size, "a.csv");
        std::cout << "Matrix A generated: a.csv" << std::endl;
        matrix_B = createRandomMatrix(N_size, N_size);
        saveMatrixToCSV(matrix_B, N_size, N_size, "b.csv");
        std::cout << "Matrix B generated: b.csv" << std::endl;
        // transpose matrix B
        std::cout << "Transposing matrix B." << std::endl;
        matrix_B_T = transposeMatrix(matrix_B, N_size);
        std::cout << "Transpose matrix B completed." << std::endl;
        // allocate matrix c
        matrix_C = new int[N_size * N_size];
    } else {
        matrix_B_T = new int[N_size * N_size];
    }

    // syncronize
    MPI_Barrier(MPI_COMM_WORLD);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // -- 3. distribute paritial matrix A to each process using Scatter
    int row_share = N_size / num_procs;
    int* matrix_A_part = new int[row_share * N_size];
    MPI_Scatter(
        matrix_A,                   // const void *sendbuf, 
        N_size * row_share,                     // int sendcount, 
        MPI_INT,                    // MPI_Datatype sendtype, 
        matrix_A_part,              // void *recvbuf, 
        N_size * row_share,                     // int recvcount, 
        MPI_INT,                    // MPI_Datatype recvtype, 
        0,                          // int root, 
        MPI_COMM_WORLD              // MPI_Comm comm
    );
    // std::cout << "Process " << rank << " receive part of matrix A: " << std::endl;
    // printMatrix(matrix_A_part, row_share, N_size);

    // syncronize
    MPI_Barrier(MPI_COMM_WORLD);

    // -- 4. broadcast matrix B_T to other process
    MPI_Bcast(
        matrix_B_T,                   // void *buffer, 
        N_size * N_size,            // int count, 
        MPI_INT,                    // MPI_Datatype datatype, 
        0,                          // int root, 
        MPI_COMM_WORLD
    );
    // std::cout << "Process " << rank << " receive matrix B_T: " << std::endl;
    // printMatrix(matrix_B_T, N_size, N_size);

    // syncronize
    MPI_Barrier(MPI_COMM_WORLD);

    // -- 5. multiplication
    int* matrix_C_part = new int[row_share * N_size];
    // std::cout << "Multiplying on rank " << rank << std::endl;
    matrix_C_part = multiplication(matrix_A_part, matrix_B_T, row_share, N_size);
    // std::cout << "Finish multiplying on rank " << rank << std::endl;
    // std::cout << "Process " << rank << " calculated matrix C part: " << std::endl;

    // syncronize
    MPI_Barrier(MPI_COMM_WORLD);

    // -- 6. gather matrix C
    MPI_Gather(
        matrix_C_part,          // const void *sendbuf, 
        row_share * N_size,     // int sendcount, 
        MPI_INT,                // MPI_Datatype sendtype, 
        matrix_C,               // void *recvbuf, 
        row_share * N_size,     // int recvcount, 
        MPI_INT,                // MPI_Datatype recvtype, 
        0,                      // int root, 
        MPI_COMM_WORLD
    );

    // syncronize
    MPI_Barrier(MPI_COMM_WORLD);

    gettimeofday(&end, NULL);

    // dellocate data
    if (rank == 0) {
        delete[] matrix_A;
        delete[] matrix_B;
        delete[] matrix_B_T;
        delete[] matrix_A_part;
        delete[] matrix_C_part;
    } else {
        delete[] matrix_A_part;
        delete[] matrix_B_T;
        delete[] matrix_C_part;
    }

    if (rank == 0) {
        saveMatrixToCSV(matrix_C, N_size, N_size, "c.csv");
        std::cout << "Multiplication completed. Result in c.csv" << std::endl;
        delete[] matrix_C;
        // printMatrix(matrix_C, N_size, N_size);
    }

    // syncronize
    MPI_Barrier(MPI_COMM_WORLD);

    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;
    printf("Parallel Execution Time on process %d: %f seconds\n", rank, elapsed);

    MPI_Finalize();

    return 0;
}