#ifndef TRIE_TREE_H
#define TRIE_TREE_H

#include <vector>
#include <map>
#include <memory>
#include "config.h"


namespace ANNS {

    // trie tree node
    struct TrieNode {
        LabelType label;
        IdxType group_id;                       // group_id>0, and 0 if not a terminal node
        LabelType label_set_size;               // number of elements in the label set if it is a terminal node
        IdxType group_size;                     // number of elements in the group if it is a terminal node

        std::shared_ptr<TrieNode> parent;
        std::map<LabelType, std::shared_ptr<TrieNode>> children;

        TrieNode(LabelType x, std::shared_ptr<TrieNode> y)
            : label(x), parent(y), group_id(0), label_set_size(0), group_size(0) {}
        TrieNode(LabelType a, IdxType b, LabelType c, IdxType d)
            : label(a), group_id(b), label_set_size(c), group_size(d) {}
        ~TrieNode() = default;
    };


    // trie tree construction and search for super sets
    class TrieIndex {

        public:
            TrieIndex();

            // construction
            IdxType insert(const std::vector<LabelType>& label_set, IdxType& new_label_set_id);

            // query
            LabelType get_max_label_id() const { return _max_label_id; }
            std::shared_ptr<TrieNode> find_exact_match(const std::vector<LabelType>& label_set) const;
            void get_super_set_entrances(const std::vector<LabelType>& label_set, 
                                         std::vector<std::shared_ptr<TrieNode>>& super_set_entrances, 
                                         bool avoid_self=false, bool need_containment=true) const;

            // I/O
            void save(std::string filename) const;
            void load(std::string filename);
            float get_index_size();

        private:
            LabelType _max_label_id = 0;
            std::shared_ptr<TrieNode> _root;
            std::vector<std::vector<std::shared_ptr<TrieNode>>> _label_to_nodes;

            // help function for get_super_set_entrances
            bool examine_smallest(const std::vector<LabelType>& label_set, const std::shared_ptr<TrieNode>& node) const;
            bool examine_containment(const std::vector<LabelType>& label_set, const std::shared_ptr<TrieNode>& node) const;
    };
}

#endif // TRIE_TREE_H