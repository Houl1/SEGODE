Orca is implemented as a simple command-line utility that consists of a single C++ source file. It has no external dependencies and requires only a relatively modern compiler that supports unordered_map from the C++11 standard, which most do.

Current GCC compiler requires a -std=c++11 flag. For example, you can compile the source using MinGW compiler on Windows with:
g++ -O2 -std=c++11 -o orca.exe orca.cpp

The utility takes three command-line arguments:
orca.exe 5 example.in example.out
1. graphlet size
This indicates the size of graphlets that you wish to count and should be either 4 or 5.
2. input file
Input file describes the network in a simple text format. The first line contains two integers n and e - the number of nodes and edges. The following e lines describe undirected edges with space-separated ids of their endpoints. Node ids should be between 0 and n-1. See graph.in as an example.
3. output file
Output file will consist of n lines, one for each node in a graph from 0 to n-1. Every line will contain 15 or 73 space-separated orbit counts depending on the specified graphlet size.
