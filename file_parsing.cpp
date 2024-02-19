#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>


std::vector<std::vector<double>> read_csv(std::string filename) {
    std::vector<std::vector<double>> table;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::string msg = "File" + filename + "won't open\n";
        throw std::runtime_error(msg);
    }

    std::string line;

    int row_number = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> row;

        if (row_number == 0) {
            row_number++;
            continue;
        }

        while (std::getline(ss, token, ',')) {
            double token_numeric = std::stod(token);
            row.push_back(token_numeric);
        }
        table.push_back(row);
    }
    return table;
}
