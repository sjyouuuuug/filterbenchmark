#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <mutex>
#include <fstream>
#include <sstream>
#include "config.h"


namespace ANNS {

    class Graph {
            
        public:
            std::vector<IdxType>* neighbors;
            std::mutex* neighbor_locks;

            Graph() = default;

            Graph(IdxType num_points) {
                _num_points = num_points;
                neighbors = new std::vector<IdxType>[num_points];
                neighbor_locks = new std::mutex[num_points];
            };

            Graph(std::shared_ptr<Graph> graph, IdxType start, IdxType end) {
                neighbors = graph->neighbors + start;
                neighbor_locks = graph->neighbor_locks + start;
                _num_points = end - start;
            };

            void save(std::string& filename) {
                std::ofstream out(filename);
                for (IdxType i = 0; i < _num_points; i++) {
                    out << i << " ";
                    for (auto& neighbor : neighbors[i])
                        out << neighbor << " ";
                    out << std::endl;
                }
                out.close();
            }

            void load(std::string& filename) {
                std::ifstream in(filename);
                std::string line;
                IdxType id, neighbor;
                while (std::getline(in, line)) {
                    std::istringstream iss(line);
                    iss >> id;
                    neighbors[id].clear();
                    while (iss >> neighbor) 
                        neighbors[id].push_back(neighbor);
                }
                in.close();
            }

            float get_index_size() {
                float index_size = 0;
                for (IdxType i = 0; i < _num_points; i++)
                    index_size += neighbors[i].size() * sizeof(IdxType);
                return index_size;
            }

            void clean() {
                delete[] neighbors;
                delete[] neighbor_locks;
            }

            ~Graph() = default;

        private:

            IdxType _num_points;
            
    };
}

#endif // GRAPH_H