// Create hostfile for MPI runs
// Get integer ids from user input, for  example: "0 1 2 3"
// Final integer is the number of iterations , for example: user inputs "0 1 2"
// Output:
// MPI-node0
// MPI-node1
// MPI-node2
// MPI-node0 (second iteration because the final ouput is 2)
// and so on.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
using namespace std;

int main() {
    cout << "Enter space-separated integer node IDs (final integer is number of iterations): ";
    string line;
    getline(cin, line);
    istringstream iss(line);
    vector<int> node_ids;
    int id;
    while (iss >> id) {
        node_ids.push_back(id);
    }

    if (node_ids.size() < 2) {
        cerr << "Please provide at least one node ID and the number of iterations." << endl;
        return 1;
    }

    int iterations = node_ids.back();
    node_ids.pop_back();

    ofstream hostfile("hostfile.txt");
    if (!hostfile.is_open()) {
        cerr << "Failed to open hostfile.txt for writing." << endl;
        return 1;
    }

    for (int iter = 0; iter < iterations; iter++) {
        for (int node_id : node_ids) {
            hostfile << "MPI-node" << node_id << endl;
        }
    }

    hostfile.close();
    cout << "Hostfile 'hostfile.txt' created successfully." << endl;
    return 0;
}