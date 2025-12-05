#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <omp.h>
#include <stdexcept>

// Global constants for clearer indexing
#define IDX(i, j, cols) ((i) * (cols) + (j))

/**
 * Creates a dynamically allocated 1D array representing a 2D matrix (int*)
 * in contiguous memory (Row-Major Order), initialized with random integers [0, 9].
 *
 * @param rows The number of rows for the matrix.
 * @param cols The number of columns for the matrix (the stride).
 * @return A pointer to the data (int*).
 */
int *createRandomMatrix(int rows, int cols)
{
    // Seed the random number generator only once per program execution
    static bool seeded = false;
    if (!seeded)
    {
        std::srand(static_cast<unsigned int>(std::time(0)));
        seeded = true;
    }

    if (rows <= 0 || cols <= 0)
    {
        return nullptr;
    }

    // Allocate memory for rows * cols elements
    int *matrix = new int[rows * cols];

    // Initialize them
    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = std::rand() % 10;
    }

    return matrix;
}

/**
 * Helper function to print the matrix.
 */
void printMatrix(int *matrix, int rows, int cols)
{
    if (!matrix)
        return;
    std::cout << "Generated Matrix (" << rows << "x" << cols << "):" << std::endl;
    // Limit printing for large matrices to keep output clean
    int limit = (rows > 10 || cols > 10) ? 5 : rows;

    for (int i = 0; i < limit; ++i)
    {
        for (int j = 0; j < limit; ++j)
        {
            // Using the correct Row-Major indexing formula
            std::cout << matrix[IDX(i, j, cols)] << " ";
        }
        if (cols > 10)
            std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > 10)
        std::cout << "..." << std::endl;
}

/**
 * Helper function to properly deallocate the dynamically created matrix.
 */
void deleteMatrix(int *matrix)
{
    delete[] matrix;
}

/**
 * Saves an int* matrix to a CSV file.
 */
bool saveMatrixToCSV(int *matrix, int rows, int cols, const std::string &filename)
{
    std::ofstream outFile(filename);

    if (!outFile.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return false;
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            outFile << matrix[IDX(i, j, cols)];

            if (j < cols - 1)
            {
                outFile << ",";
            }
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "Successfully saved matrix to " << filename << std::endl;
    return true;
}

/**
 * Creates and returns the transpose of the input matrix.
 * Note: Since the input matrix is square (size x size), the output matrix
 * will also be size x size.
 */
int *transposeMatrix(int *matrix, int size)
{
    int *matrix_T = new int[size * size];

// Transpose is safe to parallelize for large matrices
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            // matrix_T[i][j] = matrix[j][i]
            matrix_T[IDX(i, j, size)] = matrix[IDX(j, i, size)];
        }
    }
    return matrix_T;
}

/**
 * Optimized square matrix multiplication using the transpose of B.
 * This is crucial for fixing the cache-miss problem.
 * * @param matrix_A The first matrix (A).
 * @param matrix_B_T The transpose of the second matrix (B^T).
 * @param size The dimension (N) of the square matrices.
 * @return The result matrix C = A * B.
 */
int *multiplyMatrix_Optimized(int *matrix_A, int *matrix_B_T, int size)
{
    int *matrix_C = new int[size * size];
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int sum = 0;
            for (int k = 0; k < size; k++)
            {
                int a_val = matrix_A[IDX(i, k, size)];
                int b_t_val = matrix_B_T[IDX(j, k, size)];
                sum += a_val * b_t_val;
            }
            matrix_C[IDX(i, j, size)] = sum;
        }
    }
    return matrix_C;
}

int main(int argc, char **argv)
{
    // Basic argument validation (as before)
    if (argc != 2)
    {
        std::cerr << "Error: Invalid Argument Count. This program expects one size parameter." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <N_size>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 1024" << std::endl;
        return 1;
    }

    int N_size;
    try
    {
        N_size = std::stoi(argv[1]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: Invalid input for matrix size. Please enter an integer." << std::endl;
        return 1;
    }

    if (N_size <= 0)
    {
        std::cerr << "Error: Matrix size must be a positive integer." << std::endl;
        return 1;
    }

    // Check for excessive size that could lead to extremely long runtimes
    if (N_size > 5000)
    {
        std::cout << "Warning: Matrix size N=" << N_size << " is extremely large (N^3 = "
                  << static_cast<long long>(N_size) * N_size * N_size / 1000000000.0 << " trillion operations). "
                  << "This will take a very long time, even with optimization." << std::endl;
    }

    std::cout << "Matrix size accepted. All matrices will be: "
              << N_size << " x " << N_size << std::endl;

    // 1. Create original matrices A and B
    std::cout << "Allocating and initializing matrices..." << std::endl;
    int *matrix_A = createRandomMatrix(N_size, N_size);
    int *matrix_B = createRandomMatrix(N_size, N_size);

    // 2. Transpose matrix B (O(N^2) operation - very fast)
    std::cout << "Transposing matrix B for cache optimization..." << std::endl;
    double start_transpose = omp_get_wtime();
    int *matrix_B_T = transposeMatrix(matrix_B, N_size);
    double end_transpose = omp_get_wtime();
    std::cout << "Transpose time: " << (end_transpose - start_transpose) << " seconds" << std::endl;

    // 3. Perform optimized multiplication (A * B = A * B^T)
    std::cout << "Starting optimized matrix multiplication (A * B)..." << std::endl;
    double start_multiply = omp_get_wtime();
    int *matrix_C = multiplyMatrix_Optimized(matrix_A, matrix_B_T, N_size);
    double end_multiply = omp_get_wtime();

    double elapsed_time = end_multiply - start_multiply;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "Optimized Multiplication Time: " << elapsed_time << " seconds" << std::endl;
    std::cout << "Total Runtime (Transp. + Mult.): " << (end_multiply - start_transpose) << " seconds" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    // 4. Cleanup
    deleteMatrix(matrix_A);
    deleteMatrix(matrix_B); // Delete original B
    // deleteMatrix(matrix_B_T); // Delete B_T
    deleteMatrix(matrix_C);

    return 0;
}