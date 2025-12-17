#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>

int main() {
    std::cout << "Enter node IDs (e.g. 1 2 3): ";

    std::string line;
    std::getline(std::cin, line);

    std::istringstream iss(line);
    std::vector<int> ids;
    int id;

    while (iss >> id) {
        ids.push_back(id);
    }

    // GET file name from user
    std::cout << "Enter file name to transfer: ";
    std::string file;
    std::getline(std::cin, file);
    file = "./" + file;

    std::string remote_path =
        "/root/group_05/assign251_co3067/strassen/";

    for (int i : ids) {
        std::string host = "MPI-node" + std::to_string(i);

        std::string cmd =
            "scp " + file + " root@" + host + ":" + remote_path;

        std::cout << "Running: " << cmd << std::endl;

        int ret = system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "Failed on " << host << std::endl;
        }
    }

    return 0;
}
