#include <set>
#include <queue>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include "trie.h"


namespace ANNS {
    
    TrieIndex::TrieIndex() {
        _root = std::make_shared<TrieNode>(0, nullptr);
    }


    // insert a new label set into the trie tree, increase the group size
    IdxType TrieIndex::insert(const std::vector<LabelType>& label_set, IdxType& new_label_set_id) {
        std::shared_ptr<TrieNode> cur = _root;
        for (const LabelType label : label_set) {

            // create a new node
            if (cur->children.find(label) == cur->children.end()) {
                cur->children[label] = std::make_shared<TrieNode>(label, cur);

                // update max label id and label_to_nodes
                if (label > _max_label_id) {
                    _max_label_id = label;
                    _label_to_nodes.resize(_max_label_id+1);
                }
                _label_to_nodes[label].push_back(cur->children[label]);
            }
            cur = cur->children[label];
        }
        
        // set the group_id and group_size
        if (cur->group_id == 0) {
            cur->group_id = new_label_set_id++;
            cur->label_set_size = label_set.size();
            cur->group_size = 1;
        } else {
            cur->group_size++;
        }
        return cur->group_id;
    }


    // find the exact match of the label set
    std::shared_ptr<TrieNode> TrieIndex::find_exact_match(const std::vector<LabelType>& label_set) const {
        std::shared_ptr<TrieNode> cur = _root;
        for (const LabelType label : label_set) {
            if (cur->children.find(label) == cur->children.end()) 
                return nullptr;
            cur = cur->children[label];
        }

        // check whether it is a terminal node
        if (cur->group_id == 0)
            return nullptr;
        return cur;
    }


    // get the top entrances of all super sets in the trie tree, assume the label_set has been sorted in ascending order
    void TrieIndex::get_super_set_entrances(const std::vector<LabelType>& label_set,
                                            std::vector<std::shared_ptr<TrieNode>>& super_set_entrances,
                                            bool avoid_self, bool need_containment) const {
        super_set_entrances.clear();

        // find the existing node for the input label set
        std::shared_ptr<TrieNode> avoided_node = nullptr;
        if (avoid_self)
            avoided_node = find_exact_match(label_set);        
        std::queue<std::shared_ptr<TrieNode>> q;

        // if the label set is empty, find all children of the root
        if (label_set.empty()) {
            for (const auto& child : _root->children)
                q.push(child.second);
        } else {

            // if need containing the input label set, obtain candidate nodes for the last label
            if (need_containment) {
                for (auto node : _label_to_nodes[label_set[label_set.size()-1]])
                    if (examine_containment(label_set, node))
                        q.push(node);
            
            // if no need for containing the whole label set
            } else {
                for (auto label : label_set)
                    for (auto node : _label_to_nodes[label])
                        if (examine_smallest(label_set, node))
                            q.push(node);
            }
        }

        // search in the trie tree to find the candidate super sets
        std::set<IdxType> group_ids;
        while (!q.empty()) {
            auto cur = q.front();
            q.pop();

            // add to candidates if it is a terminal node
            if (cur->group_id > 0 && cur != avoided_node && group_ids.find(cur->group_id) == group_ids.end()) {
                group_ids.insert(cur->group_id);
                super_set_entrances.push_back(cur);
            } else {
                for (const auto& child : cur->children)
                    q.push(child.second);
            }
        }
    }


    // bottom to top, examine whether the current node is the smallest in the label set
    bool TrieIndex::examine_smallest(const std::vector<LabelType>& label_set, 
                                     const std::shared_ptr<TrieNode>& node) const {             
        auto cur = node->parent;
        while (cur != nullptr && cur->label >= label_set[0]) {
            if (std::binary_search(label_set.begin(), label_set.end(), cur->label))
                return false;
            cur = cur->parent;
        }
        return true;
    }


    // bottom to top, examine whether is a super set of the label set
    bool TrieIndex::examine_containment(const std::vector<LabelType>& label_set, 
                                      const std::shared_ptr<TrieNode>& node) const {
        auto cur = node->parent;
        for (int64_t i = label_set.size()-2; i>=0; --i) {
            while (cur->label > label_set[i] && cur->parent != nullptr)
                cur = cur->parent;
            if (cur->parent == nullptr || cur->label != label_set[i])
                return false;
        }
        return true;
    }


    // save the trie tree to a file
    void TrieIndex::save(std::string filename) const {
        std::ofstream out(filename);

        // save the max label id and number of nodes
        out << _max_label_id << std::endl;
        IdxType num_nodes = 1;
        for (const auto& nodes : _label_to_nodes)
            num_nodes += nodes.size();
        out << num_nodes << std::endl;

        // save the root node
        std::unordered_map<std::shared_ptr<TrieNode>, IdxType> node_to_id;
        out << 0 << " " << _root->label << " " << _root->group_id << " " \
            << _root->label_set_size << " " << _root->group_size << std::endl;
        node_to_id[_root] = 0;
        
        // save the other nodes
        IdxType id = 1;
        for (const auto& nodes : _label_to_nodes)
            for (const auto& node : nodes) {
                out << id << " " << node->label << " " << node->group_id << " " \
                << node->label_set_size << " " << node->group_size << std::endl;
                node_to_id[node] = id;
                ++id;
            }

        // save the parent of each node
        for (const auto& each : node_to_id) {
            if (each.first == _root)
                out << each.second << " 0" << std::endl;
            else
                out << each.second << " " << node_to_id[each.first->parent] << std::endl;
        }

        // save the children of each node
        for (const auto& each : node_to_id) {
            out << each.second << " " << each.first->children.size() << " ";
            for (const auto& child : each.first->children)
                out << child.first << " " << node_to_id[child.second] << " ";
            out << std::endl;
        }
    }


    // load the trie tree from a file
    void TrieIndex::load(std::string filename) {
        std::ifstream in(filename);
        LabelType label, num_children, label_set_size;
        IdxType id, group_id, group_size, parent_id, child_id;

        // load the max label id and number of nodes
        in >> _max_label_id;
        IdxType num_nodes;
        in >> num_nodes;

        // load the nodes
        std::vector<std::shared_ptr<TrieNode>> nodes(num_nodes);
        for (IdxType i=0; i<num_nodes; ++i) {
            in >> id >> label >> group_id >> label_set_size >> group_size;
            nodes[id] = std::make_shared<TrieNode>(label, group_id, label_set_size, group_size);
        }
        _root = nodes[0];

        // load the parent of each node
        for (IdxType i=0; i<num_nodes; ++i) {
            in >> id >> parent_id;
            if (id > 0)
                nodes[id]->parent = nodes[parent_id];
        }
        _root->parent = nullptr;

        // load the children of each node
        for (IdxType i=0; i<num_nodes; ++i) {
            in >> id >> num_children;
            for (IdxType j=0; j<num_children; ++j) {
                in >> label >> child_id;
                nodes[id]->children[label] = nodes[child_id];
            }
        }

        // build label_to_nodes
        _label_to_nodes.resize(_max_label_id+1);
        for (const auto each : nodes)
            _label_to_nodes[each->label].push_back(each);
    }



    float TrieIndex::get_index_size() {
        float index_size = 0;
        for (const auto& nodes : _label_to_nodes) {
            index_size += nodes.size() * ( sizeof(TrieNode) + sizeof(std::shared_ptr<TrieNode>) );
            for (const auto& node : nodes)
                index_size += node->children.size() * ( sizeof(LabelType) + sizeof(std::shared_ptr<TrieNode>) );
        }
        return index_size;
    }
}
