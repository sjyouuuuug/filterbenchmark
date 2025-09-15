#ifndef LABEL_NAV_GRAPH_H
#define LABEL_NAV_GRAPH_H

#include <vector>
#include "config.h"


namespace ANNS {

    class LabelNavGraph {

        public:
            LabelNavGraph(IdxType num_nodes) {
                in_neighbors.resize(num_nodes+1);
                out_neighbors.resize(num_nodes+1);
            };

            std::vector<std::vector<IdxType>> in_neighbors, out_neighbors;

            ~LabelNavGraph() = default;

        private:
            
    };
}



#endif // LABEL_NAV_GRAPH_H