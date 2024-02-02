//
// Created by rando on 1/29/24.
//

#include <cmath>
//#include <cstdio>
#include <cstring>
#include<limits>

#include "DecisionTreeNode.h"

namespace pico_dt {
    DecisionTreeNode::DecisionTreeNode(size_t p_parameter_count, int p_label_count) {
        parameter_count = p_parameter_count;
        label_count = p_label_count;
        default_value = 0;
        lesser_branch = nullptr;
        greater_branch = nullptr;
        parent_branch = nullptr;
        comparison_parameter = -1;
        comparison_threshold = -1.0;
    }

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"

    void DecisionTreeNode::fit(double **parameters, int *labels, size_t count) {
        // TODO: make this non-recursive, probably using parent_branch like in serialize
        //  because apparently doing this recursively is bad
        //  (technically it can cause a stack overflow)
        //  storing a full parameter stack should be avoided
        //  the parameter stack for a node can be recalculated quickly by the following
        //  1. parameter list starts as full parameter list
        //  2. current node starts as this node
        //  3. if current node is lesser branch of parent branch
        //     3.a. remove all samples greater than parent branch threshold from parameter list
        //  4. if current node is lesser branch of parent branch
        //     4.a. remove all samples less than parent branch threshold from parameter list
        //  5. if parent branch is null, stopp
        //  6. goto 3
        //  This is basically the reverse of what happens in this method when going down the tree.
        //  proof: A & B & C == C & B & A

        //printf("Starting fit of tree, %zu parameters sent.\n", count);
        if (calculate_entropy(labels, count) <= 0) {
            //printf("Node is a leaf!\n");
            // the entropy is 0! This node is pure! Find the first label and set it as the default value.
            default_value = labels[0];
            return;
        }
        //printf("Node isn't a leaf.\n");

        // create list of test splits
        auto **test_splits = new double *[parameter_count];
        for (size_t i = 0; i < parameter_count; ++i) {
            // create a list for the test splits of this parameter
            test_splits[i] = new double[count];
            for (size_t j = 0; j < count; ++j) {
                test_splits[i][j] = -std::numeric_limits<double>::infinity();
            }

            // add all of the copies of this parameter to the array
            for (size_t j = 0; j < count; j++) {
                double val = parameters[j][i];

                // insertion sort it into the array.
                for (size_t k = 0; k < count; k++) {
                    if (val > test_splits[i][k]) {
                        double temp = val;
                        val = test_splits[i][k];
                        test_splits[i][k] = temp;
                    }
                }
            }

            // finalize the array values
            for (size_t j = 0; j < count - 1; ++j) {
                // going to try to use the midpoints between values, not the values themselves.
                test_splits[i][j] = (test_splits[i][j] + test_splits[i][j + 1]) / 2;
            }
        }

        // okay, we've picked our test parameters, now to evaluate.
        size_t best_parameter = 0;
        size_t best_split = 0;
        double best_score = calculate_information_gain(parameters, labels, count, 0,
                                                       test_splits[0][0]);
        for (size_t i = 0; i < parameter_count; ++i) {
            for (size_t j = 0; j < count - 1; ++j) {
                double this_score = calculate_information_gain(parameters, labels, count, i,
                                                               test_splits[i][j]);
                //printf("%zu, %zu, %lf, %lf\n", i, j, test_splits[i][j], this_information_gain_ratio);
                // use >=, not > as lower values are sorted later
                // in the case of parameters like 2.0, 2.0, 0.0, 0.0, the splits will be 2.0, 1.0, 0.0
                // both 2.0 and 1.0 will split the samples in half, but 1.0 will split down the middle of the
                // gap between parameters, not right at one on the parameters
                if (this_score >= best_score) {
                    best_score = this_score;
                    best_parameter = i;
                    best_split = j;
                }
            }
        }

        comparison_parameter = best_parameter;
        comparison_threshold = test_splits[best_parameter][best_split];

        // cleanup. instead of waiting until the very end, we'll clean up now before recursing.
        for (size_t i = 0; i < parameter_count; ++i) {
            delete[] test_splits[i];
        }
        delete[] test_splits;

        size_t lesser_count = 0;
        for (size_t i = 0; i < count; i++) {
            if (parameters[i][comparison_parameter] < comparison_threshold) ++lesser_count;
        }
        size_t greater_count = count - lesser_count;

        // recursively fit the lesser child
        //printf("Fitting lesser child.\n");
        auto **lesser_parameters = new double *[lesser_count];
        auto *lesser_labels = new int[lesser_count];
        size_t lesser_index = 0;
        for (size_t parent_index = 0; parent_index < count; parent_index++) {
            if (parameters[parent_index][comparison_parameter] >= comparison_threshold) continue;
            lesser_parameters[lesser_index] = parameters[parent_index];
            lesser_labels[lesser_index++] = labels[parent_index];
        }
        lesser_branch = new DecisionTreeNode(parameter_count, label_count);
        lesser_branch->parent_branch = this;
        lesser_branch->fit(lesser_parameters, lesser_labels, lesser_count);
        //printf("Done fitting lesser child.\n");
        delete[] lesser_parameters;
        delete[] lesser_labels;

        // recursively fit the greater child
        //printf("Fitting greater child.\n");
        auto **greater_parameters = new double *[greater_count];
        auto *greater_labels = new int[greater_count];
        size_t greater_index = 0;
        for (size_t parent_index = 0; parent_index < count; parent_index++) {
            if (parameters[parent_index][comparison_parameter] < comparison_threshold) continue;
            greater_parameters[greater_index] = parameters[parent_index];
            greater_labels[greater_index++] = labels[parent_index];
        }
        greater_branch = new DecisionTreeNode(parameter_count, label_count);
        greater_branch->parent_branch = this;
        greater_branch->fit(greater_parameters, greater_labels, greater_count);
        //printf("Done fitting greater child.\n");
        delete[] greater_parameters;
        delete[] greater_labels;

    }

#pragma clang diagnostic pop

    double DecisionTreeNode::calculate_entropy(const int *labels, size_t total_count) const {

        // count how many of each label there are.
        auto *label_counts = new size_t[label_count];
        for (size_t i = 0; i < label_count; ++i) {
            label_counts[i] = 0;
        }
        for (size_t i = 0; i < total_count; i++) {
            int label = labels[i];
            ++label_counts[label];
        }

        // calculate the entropy
        double entropy = 0.0;
        for (int i = 0; i < label_count; ++i) {  // run the summation
            double p_i = (double) label_counts[i] / (double) total_count;
            if (p_i == 0) continue;
            entropy += p_i * log2(p_i);
        }
        entropy = -entropy;

        delete[] label_counts;
        return entropy;
    }

    double
    DecisionTreeNode::calculate_information_gain(double **parameters, const int *labels, size_t total_count,
                                                 size_t split_parameter, double split_threshold) const {
        // count how many of each label there are.
        auto *parent_label_counts = new size_t[label_count];
        auto *lesser_child_label_counts = new size_t[label_count];
        auto *greater_child_label_counts = new size_t[label_count];
        for (size_t i = 0; i < label_count; ++i) {
            parent_label_counts[i] = 0;
            lesser_child_label_counts[i] = 0;
            greater_child_label_counts[i] = 0;
        }
        double lesser_child_split_count = 0;
        double greater_child_split_count = 0;
        for (size_t i = 0; i < total_count; i++) {

            //add to the total label count (parent entropy)
            int label = labels[i];
            ++parent_label_counts[label];

            // add to the child label counts
            if (parameters[i][split_parameter] < split_threshold) {
                ++lesser_child_label_counts[label];
                ++lesser_child_split_count;
            } else {
                ++greater_child_label_counts[label];
                ++greater_child_split_count;
            }
        }

        if (lesser_child_split_count <= 0 || greater_child_split_count <= 0) return 0;

        // calculate the parent entropy
        double parent_entropy = 0.0;
        for (int i = 0; i < label_count; ++i) {  // run the summation
            double p_i = (double) parent_label_counts[i] / (double) total_count;
            if (p_i == 0) continue;
            parent_entropy += p_i * log2(p_i);
        }
        parent_entropy = -parent_entropy;

        //calculate the child entropies
        double lesser_child_entropy = 0.0;
        for (int i = 0; i < label_count; ++i) {  // run the summation
            double p_i = (double) lesser_child_label_counts[i] / (double) lesser_child_split_count;
            if (p_i == 0) continue;
            lesser_child_entropy += p_i * log2(p_i);
        }
        lesser_child_entropy = -lesser_child_entropy;
        double greater_child_entropy = 0.0;
        for (int i = 0; i < label_count; ++i) {  // run the summation
            double p_i = (double) greater_child_label_counts[i] / (double) greater_child_split_count;
            if (p_i == 0) continue;
            greater_child_entropy += p_i * log2(p_i);
        }
        greater_child_entropy = -greater_child_entropy;

        delete[] parent_label_counts;
        delete[] lesser_child_label_counts;
        delete[] greater_child_label_counts;
        return parent_entropy - ((lesser_child_entropy + greater_child_entropy) / 2);
    }

    int DecisionTreeNode::predict(const double *parameters) {
        if (lesser_branch == nullptr || greater_branch == nullptr) return default_value;
        DecisionTreeNode *dtn;
        if (parameters[comparison_parameter] < comparison_threshold) dtn = lesser_branch;
        else dtn = greater_branch;

        while (true) {
            if (dtn->lesser_branch == nullptr || dtn->greater_branch == nullptr) return dtn->default_value;
            if (parameters[dtn->comparison_parameter] < dtn->comparison_threshold) dtn = dtn->lesser_branch;
            else dtn = dtn->greater_branch;
        }
    }

    DecisionTreeNode::DecisionTreeNode(size_t p_parameter_count, int p_label_count, int p_default_value) {
        //printf("Creating decision tree node %p with default value of %d\n", this, p_default_value);
        parameter_count = p_parameter_count;
        label_count = p_label_count;
        default_value = p_default_value;
        lesser_branch = nullptr;
        greater_branch = nullptr;
        parent_branch = nullptr;
        comparison_parameter = -1;
        comparison_threshold = -1.0;
    }

    DecisionTreeNode::DecisionTreeNode(size_t p_parameter_count, int p_label_count, size_t p_comparison_parameter,
                                       double p_comparison_threshold, DecisionTreeNode *p_lesser_branch,
                                       DecisionTreeNode *p_greater_branch) {
        //printf("Creating decision tree node %p with comparison threshold of %lf on parameter %ld, pointing to %p and %p\n", this, p_comparison_threshold, p_comparison_parameter, p_lesser_branch, p_greater_branch);
        parameter_count = p_parameter_count;
        label_count = p_label_count;
        default_value = 0;
        lesser_branch = p_lesser_branch;
        lesser_branch->parent_branch = this;
        greater_branch = p_greater_branch;
        greater_branch->parent_branch = this;
        comparison_parameter = p_comparison_parameter;
        comparison_threshold = p_comparison_threshold;
    }

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"

    size_t DecisionTreeNode::calculate_serialized_size() {
        if (lesser_branch == nullptr || greater_branch == nullptr) return 1 + sizeof(default_value);
        return 1 + sizeof(comparison_parameter) + sizeof(comparison_threshold) +
               lesser_branch->calculate_serialized_size() + greater_branch->calculate_serialized_size();
    }

#pragma clang diagnostic pop

    void DecisionTreeNode::serialize_leaf(uint8_t *location) {
        location[0] = PICO_DT_LEAF_FLAG;
        memcpy(location + 1, &default_value, sizeof(default_value));
    }

    void DecisionTreeNode::serialize_branch(uint8_t *location) {
        location[0] = PICO_DT_BRANCH_FLAG;
        memcpy(location + 1, &comparison_parameter, sizeof(comparison_parameter));
        memcpy(location + 1 + sizeof(comparison_parameter), &comparison_threshold, sizeof(comparison_threshold));
    }

    uint8_t *DecisionTreeNode::serialize() {
        auto *buffer = new uint8_t[calculate_serialized_size()];
        auto *buffer_location = buffer;
        DecisionTreeNode *next_node = this;
        while (true) {
            //printf("DTN: %p, LN: %p, GN: %p\n", next_node, next_node->lesser_branch, next_node->greater_branch);
            if (next_node->lesser_branch != nullptr || next_node->greater_branch != nullptr) {
                next_node = next_node->lesser_branch;
                continue;
            }

            //printf("Serialize leaf node: %p\n", next_node);
            next_node->serialize_leaf(buffer_location);
            buffer_location += 1 + sizeof(default_value);

            while (true) {
                if (next_node->parent_branch == nullptr) break;
                if (next_node != next_node->parent_branch->greater_branch) {
                    next_node = next_node->parent_branch->greater_branch;
                    break;
                }
                next_node = next_node->parent_branch;
                //printf("Serialize branch node: %p\n", next_node);
                next_node->serialize_branch(buffer_location);
                buffer_location += 1 + sizeof(comparison_parameter) + sizeof(comparison_threshold);
            }
            if (next_node->parent_branch == nullptr) break;
        }

        return buffer;
    }

#pragma clang diagnostic push
#pragma ide diagnostic ignored "NullDereference"  // null dereferences are a know, and desired, effect.

    DecisionTreeNode *
    deserialize_decision_tree(size_t parameter_count, int label_count, const uint8_t *buffer, size_t buffer_length) {
        DecisionTreeNode *dt_stack = nullptr;
        for (size_t buffer_pointer = 0; buffer_pointer < buffer_length; ++buffer_pointer) {
            int default_value;
            size_t comparison_parameter;
            double comparison_threshold;
            DecisionTreeNode *new_node;
            switch (buffer[buffer_pointer]) {
                case PICO_DT_LEAF_FLAG:
                    memcpy(&default_value, buffer + buffer_pointer + 1, sizeof(default_value));
                    buffer_pointer += sizeof(default_value);
                    new_node = new DecisionTreeNode(parameter_count, label_count, default_value);
                    new_node->parent_branch = dt_stack;
                    dt_stack = new_node;
                    break;
                case PICO_DT_BRANCH_FLAG:
                    memcpy(&comparison_parameter, buffer + buffer_pointer + 1, sizeof(comparison_parameter));
                    buffer_pointer += sizeof(comparison_parameter);
                    memcpy(&comparison_threshold, buffer + buffer_pointer + 1, sizeof(comparison_threshold));
                    buffer_pointer += sizeof(comparison_threshold);
                    DecisionTreeNode *greater_branch = dt_stack;
                    dt_stack = greater_branch->parent_branch;
                    DecisionTreeNode *lesser_branch = dt_stack;
                    dt_stack = lesser_branch->parent_branch;
                    new_node = new DecisionTreeNode(parameter_count, label_count, comparison_parameter,
                                                    comparison_threshold, lesser_branch,
                                                    greater_branch);
                    new_node->parent_branch = dt_stack;
                    dt_stack = new_node;
                    break;
            }
        }
        DecisionTreeNode *final = dt_stack;
        return final;
    }

#pragma clang diagnostic pop

#ifdef PICO_DT_ENABLE_LOW_USE_FEATURES
    [[maybe_unused]] int DecisionTreeNode::predict_simple(double *parameters) {
            if (lesser_branch == nullptr || greater_branch == nullptr) return default_value;
            if (parameters[comparison_parameter] < comparison_threshold) return lesser_branch->predict_simple(parameters);
            else return greater_branch->predict_simple(parameters);
        }

    [[maybe_unused]] double
        DecisionTreeNode::calculate_split_information(double **parameters, size_t total_count, size_t split_parameter,
                                                      double split_threshold) {
            // count how many of each label there are.
            double lesser_child_split_count = 0;
            double greater_child_split_count = 0;
            for (size_t i = 0; i < total_count; i++) {

                // add to the child label counts
                if (parameters[i][split_parameter] < split_threshold) ++lesser_child_split_count;
                else ++greater_child_split_count;
            }

            // calculate the split information
            double split_information = -((((double) lesser_child_split_count / (double) total_count) *
                                          log2((double) lesser_child_split_count / (double) total_count)) +
                                         (((double) greater_child_split_count / (double) total_count) *
                                          log2((double) greater_child_split_count / (double) total_count)));

            return split_information;
        }

        [[maybe_unused]] double
        DecisionTreeNode::calculate_information_gain_ratio(double **parameters, const int *labels, size_t total_count,
                                                           size_t split_parameter, double split_threshold) const {
            // count how many of each label there are.
            auto *parent_label_counts = new size_t[label_count];
            auto *lesser_child_label_counts = new size_t[label_count];
            auto *greater_child_label_counts = new size_t[label_count];
            for (size_t i = 0; i < label_count; ++i){
                parent_label_counts[i] = 0;
                lesser_child_label_counts[i] = 0;
                greater_child_label_counts[i] = 0;
            }
            double lesser_child_split_count = 0;
            double greater_child_split_count = 0;
            for (size_t i = 0; i < total_count; i++) {

                //add to the total label count (parent entropy)
                int label = labels[i];
                ++parent_label_counts[label];

                // add to the child label counts
                if (parameters[i][split_parameter] < split_threshold) {
                    ++lesser_child_label_counts[label];
                    ++lesser_child_split_count;
                } else {
                    ++greater_child_label_counts[label];
                    ++greater_child_split_count;
                }
            }

            // calculate the parent entropy
            double parent_entropy = 0.0;
            for (int i = 0; i < label_count; ++i) {  // run the summation
                double p_i = (double) parent_label_counts[i] / (double) total_count;
                if (p_i == 0) continue;
                parent_entropy += p_i * log2(p_i);
            }
            parent_entropy = -parent_entropy;

            //calculate the child entropies
            double lesser_child_entropy = 0.0;
            for (int i = 0; i < label_count; ++i) {  // run the summation
                double p_i = (double) lesser_child_label_counts[i] / (double) lesser_child_split_count;
                if (p_i == 0) continue;
                lesser_child_entropy += p_i * log2(p_i);
            }
            lesser_child_entropy = -lesser_child_entropy;
            double greater_child_entropy = 0.0;
            for (int i = 0; i < label_count; ++i) {  // run the summation
                double p_i = (double) greater_child_label_counts[i] / (double) greater_child_split_count;
                if (p_i == 0) continue;
                greater_child_entropy += p_i * log2(p_i);
            }
            greater_child_entropy = -greater_child_entropy;

            // calculate information gain
            double information_gain = parent_entropy - ((lesser_child_entropy + greater_child_entropy) / 2);

            // calculate the split information
            double greater_child_split_info =
                    greater_child_split_count == 0 ? 0 : (((double) greater_child_split_count / (double) total_count) *
                                                          log2((double) greater_child_split_count / (double) total_count));
            double lesser_child_split_info =
                    lesser_child_split_count == 0 ? 0 : (((double) lesser_child_split_count / (double) total_count) *
                                                         log2((double) lesser_child_split_count / (double) total_count));
            double split_information = -(lesser_child_split_info + greater_child_split_info);

            delete[] parent_label_counts;
            delete[] lesser_child_label_counts;
            delete[] greater_child_label_counts;
            return split_information == 0 ? 0 : information_gain / split_information;
        }
#endif

} // pico_dt