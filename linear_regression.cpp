#include "file_parsing.h"
#include <vector>
#include <cstdlib>

class LinearRegression {
public:
    LinearRegression(double eta, int n_epoch);
    std::vector<double> params;
    double eta;
    int n_epoch;
    void fit(std::vector<std::vector<double>> &X, std::vector<double> &y);
    std::vector<double> predict(std::vector<std::vector<double>> &X);

private:
    std::vector<std::vector<double>> transpose_matrix(std::vector<std::vector<double>> &X);
    double dot_product(std::vector<std::vector<double>> &X_T, int col_no);
    double get_average_error(std::vector<double> &y_hat, std::vector<double> &y);
    void adjust_weights(std::vector<double> &y_hat, std::vector<double> &y, std::vector<std::vector<double>> &X);
};

LinearRegression::LinearRegression(double eta, int n_epoch)
    :eta(eta), n_epoch(n_epoch){}

void LinearRegression::fit(std::vector<std::vector<double>> &X, std::vector<double> &y) {
    for (int i = 0; i < X[0].size(); i++)
        params.push_back(std::rand());

    std::vector<std::vector<double>> X_T = transpose_matrix(X);
    std::vector<double> predictions;

    for (int epoch = 0; epoch < n_epoch; epoch++) {
       for (int col_no = 0; col_no < X_T[0].size(); col_no++) {
            double curr_dot_product = dot_product(X_T, col_no);
            predictions.push_back(curr_dot_product);
        }
        adjust_weights(predictions, y, X);

        predictions.clear();
    }
}

std::vector<double> LinearRegression::predict(std::vector<std::vector<double>> &X) {
    std::vector<std::vector<double>> X_T = transpose_matrix(X);
    std::vector<double> y_hat;
    for (int i = 0; i < X.size(); i++) {
        y_hat.push_back(dot_product(X_T, i));
    }
    return y_hat;
}

double LinearRegression::dot_product(std::vector<std::vector<double>> &X_T, int col_no) {
    double total = 0;
    for (int i = 0; i < X_T.size(); i++) {
        total += X_T[i][col_no] * params[i];
    }
    return total;
}

std::vector<std::vector<double>> LinearRegression::transpose_matrix(std::vector<std::vector<double>> &X) {
    std::vector<std::vector<double>> trans_matrix;

    for (int j = 0; j < X[0].size(); j++) {
        std::vector<double> x_T;
        for (int i = 0; i < X.size(); i++) {
            x_T.push_back(X[i][j]);
        }
        trans_matrix.push_back(x_T);
    }

    return trans_matrix;
}

void LinearRegression::adjust_weights(
        std::vector<double> &y_hat,
        std::vector<double> &y,
        std::vector<std::vector<double>> &X) {
    for (int i = 0; i < y_hat.size(); i++) {
        for (int j = 0; j < X[0].size(); j++) {
            double delta_theta_j = -eta * (y_hat[i] - y[i]) * X[i][j];
            params[j] += delta_theta_j;
        }
    }
}

std::vector<std::vector<double>> extract_X(std::vector<std::vector<double>> total_data) {
    std::vector<std::vector<double>> X;
    for (int i = 0; i < total_data.size(); i++) {
        std::vector<double> x;
        for (int j = 0; j < total_data[i].size() - 1; j++) {
            x.push_back(total_data[i][j]);
        }
        X.push_back(x);
    }
    return X;
}

std::vector<double> extract_y(std::vector<std::vector<double>> total_data) {
    std::vector<double> y;
    int y_index = total_data[0].size() - 1;
    for (int i = 0; i < total_data.size(); i++) {
        y.push_back(total_data[i][y_index]);
    }
    return y;
}


int main (int argc, char *argv[]) {
    std::string filename = argv[1];
    std::vector<std::vector<double>> data = read_csv(filename);
    std::vector<double> y = extract_y(data);
    std::vector<std::vector<double>> X = extract_X(data);
    int n_epochs = 1000;
    double eta = 0.0001;
    LinearRegression model(eta, n_epochs);
    model.fit(X, y);
    return 0;
}
