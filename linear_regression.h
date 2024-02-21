#include "file_parsing.h"
#include <vector>
#include <string>
#include <cstdlib>

class LinearRegression {
public:
    LinearRegression(double eta, double bias, int n_epoch, int batch_no);
    std::vector<double> params;
    double eta;
    double bias;
    int n_epoch;
    int batch_no;
    void fit(std::vector<std::vector<double>> X, std::vector<double> &y);
    std::vector<double> predict(std::vector<std::vector<double>> X);
    std::vector<std::vector<double>> transverse_matrix(std::vector<std::vector<double>> X);

private:
    double dot_product(std::vector<std::vector<double>> &X_T, int col_no);
    double get_average_error(std::vector<double> &y_hat, std::vector<double> &y);
    void adjust_weights(std::vector<double> &y_hat, std::vector<double> &y, std::vector<std::vector<double>> &X);
};
