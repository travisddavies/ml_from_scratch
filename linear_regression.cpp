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
    void fit(std::vector<std::vector<double>> &X, std::vector<double> &y);
    std::vector<std::vector<double>> predict();

private:
    std::vector<std::vector<double>> transverse_matrix(std::vector<std::vector<double>> &X);
    double dot_product(std::vector<std::vector<double>> &X_T, int col_no);
    double get_average_error(std::vector<double> &y_hat, std::vector<double> &y);
    void adjust_weights(std::vector<double> &y_hat, std::vector<double> &y, std::vector<std::vector<double>> X_T);
};

LinearRegression::LinearRegression(double eta,double bias,int n_epoch, int batch_no)
    :eta(eta), bias(bias), n_epoch(n_epoch), batch_no(batch_no) {}

void LinearRegression::fit(std::vector<std::vector<double>> &X, std::vector<double> &y) {
    for (int i = 0; i < X[0].size(); i++)
        params.push_back(std::rand());

    std::vector<std::vector<double>> X_T = transverse_matrix(X);
    std::vector<double> predictions;
    for (int epoch = 0; epoch < n_epoch; epoch++) {
        for (int col_no = 0; col_no < X_T[0].size(); col_no++) {
            double curr_dot_product = dot_product(X_T, col_no);
            predictions.push_back(curr_dot_product);
        }
        adjust_weights(predictions, y, X_T);
    }
}

double LinearRegression::dot_product(std::vector<std::vector<double>> &X_T, int col_no) {
    double total = 0;
    for (int i = 0; i < X_T.size(); i++) {
        total += X_T[i][col_no] * params[i];
    }
    return total;
}

std::vector<std::vector<double>> transverse_matrix(std::vector<std::vector<double>> &X) {
    std::vector<std::vector<double>> trans_matrix;
    for (int i = X.size() - 1; i >= 0; i--) {
        trans_matrix.push_back(X[i]);
    }

    for (int i = 0; i < trans_matrix.size(); i++) {
        for (int j = 0; i < trans_matrix[i].size(); j++) {
            if (i == j) continue;
            double temp = trans_matrix[i][j];
            trans_matrix[i][j] = trans_matrix[j][i];
            trans_matrix[j][i] = trans_matrix[i][j];
        }
    }
    return trans_matrix;
}

void LinearRegression::adjust_weights(
        std::vector<double> &y_hat,
        std::vector<double> &y,
        std::vector<std::vector<double>> X_T) {

}

int main (int argc, char *argv[]) {
    std::string filename = argv[1];
    std::vector<std::vector<double>> data = read_csv(filename);
    return 0;
}
