#include <iostream>
#include <vector>
#include <chrono>

// Function to calculate the determinant of a square matrix using cofactor expansion
double determinant(std::vector<std::vector<double>>& matrix) {
    int n = matrix.size(); // Assuming the matrix is n x n

    if (n == 1) {
        // Base case: If the matrix is 1x1, return its only element as the determinant
        return matrix[0][0];
    } else if (n == 2) {
        // Base case: If the matrix is 2x2, return the determinant using the formula ad - bc
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    } else {
        double det = 0.0;
        for (int i = 0; i < n; ++i) {
            // Calculate the cofactor of matrix[0][i]
            std::vector<std::vector<double>> cofactor_matrix(n - 1, std::vector<double>(n - 1, 0.0));
            for (int j = 1; j < n; ++j) {
                for (int k = 0, l = 0; k < n; ++k) {
                    if (k != i) {
                        cofactor_matrix[j - 1][l++] = matrix[j][k];
                    }
                }
            }
            
            // Add the cofactor to the determinant with appropriate sign
            det += matrix[0][i] * determinant(cofactor_matrix) * (i % 2 == 0 ? 1 : -1);
        }
        return det;
    }
}

int main() {
    // Example usage:
    std::vector<std::vector<double>> matrix = {{1, 2, 3, 4},
                                                {4, 5, 6, 5},
                                                {7, 8, 9, 100}, 
                                                {1, 2, 1, 4}};

    auto start_time = std::chrono::high_resolution_clock::now(); // Start timing
    double det = determinant(matrix);
    auto end_time = std::chrono::high_resolution_clock::now(); // End timing

    std::cout << "Determinant: " << det << std::endl;

    // Calculate and print the elapsed time in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Calculation time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
