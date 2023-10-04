#include <iostream>
#include <vector>
#include <cstdlib> // for rand()
#include <ctime> 

class Matrix{
    private:
        std::vector<std::vector<double>> data;
        size_t rows;
        size_t cols;
    public:
        //zero matrix 
        Matrix(size_t numRows, size_t numCols) : rows(numRows), cols(numCols) {
            data.resize(numRows, std::vector<double>(numCols, 0.0));
        }
        //identity matrix
        Matrix(size_t size) : rows(size), cols(size) {
            data.resize(size, std::vector<double>(size, 0.0));
            for (size_t i = 0; i < size; ++i) {
                data[i][i] = 1.0; // Set diagonal elements to 1
            }
        }
        // generate random n by n matrix with entries int 1 - 10
        Matrix randomSquare(int n){
            Matrix random(n);
            for (size_t i = 0; i<random.cols; i++){
                for (size_t j = 0; j<random.cols; j++){
                    random(i, j) = rand()%10;
                }
            }
            return random;
        }
        size_t numRows() const { return rows; }
        size_t numCols() const { return cols; }
        double& operator()(size_t i, size_t j) {
            return data[i][j];
        }

        const double& operator()(size_t i, size_t j) const {
            return data[i][j];
        }
        //Matrix addition
        Matrix operator+(const Matrix& other) const {
            if (rows != other.rows || cols != other.cols) {
                throw std::runtime_error("Matrix dimensions must match for addition.");
            }

            Matrix result(rows, cols);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = data[i][j] + other(i, j);
                }
            }
            return result;
        }
        // Matrix multiplication
        Matrix operator*(const Matrix& other) const {
            if (cols != other.rows) {
                throw std::runtime_error("Matrix dimensions are incompatible for multiplication.");
            }

            Matrix result(rows, other.cols);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < other.cols; ++j) {
                    for (size_t k = 0; k < cols; ++k) {
                        result(i, j) += data[i][k] * other(k, j);
                    }
                }
            }
            return result;
        }
        //Matrix by scalar multiplication
        Matrix operator*(double scalar) const{
            Matrix result(rows , cols);
            for (size_t i =0 ; i < rows ; ++i){
                for (size_t j = 0; j<cols; ++j){
                    result(i, j) = data[i][j]*scalar;
                }
            }
            return result;
        }
        //Matrix transpose
        Matrix transpose() const {
            Matrix result(cols, rows);
            for (size_t i = 0; i<cols; ++i){
                for (size_t j = 0; j<rows; ++j){
                    result(i,j)=data[j][i];
                }
            }
            return result;
        }
        //print matrix
        void print() const {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    std::cout << data[i][j] << "\t";
                }
                std::cout << std::endl;
            }
            }
        Matrix cut(std::vector<int> position){
            Matrix other(rows -1, cols - 1);
            for (size_t i = 0; i<other.rows; i++){
                if (i == position[0]){ i+=1;}
                for (size_t j = 0; j<other.cols; j++){
                    if (j == position[1]){j+=1;}
                    other(i,j) = data[i][j];
                }
            }
            return other;
        }
        // Calculate matrix determinant
        double determinant() const {
            if (rows!= cols){
                throw std::runtime_error("Cannot calculate determinant of a non-square matrix");
            }
            if (rows == 1){
                return data[0][0];
            }
            double det = 0;
            if (rows == 2){
                det = det + data[0][0]*data[1][1] - data[0][1]*data[1][0];
            }
            return det;
        }
    };
int main(){
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    Matrix I(4);
    I = I.randomSquare(4);
    I.print();
    I = I.cut({1, 1});
    I.print();
    return 0;
}