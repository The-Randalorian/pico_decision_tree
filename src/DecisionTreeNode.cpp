//
// Created by rando on 1/29/24.
//

#include <cmath>
//#include <cstdio>
#include "DecisionTreeNode.h"

namespace pico_dt {
    DecisionTreeNode::DecisionTreeNode(size_t p_parameter_count, int p_label_count) {
        parameter_count = p_parameter_count;
        label_count = p_label_count;
        default_value = 0.0;
        lesser_branch = nullptr;
        greater_branch = nullptr;
        comparison_parameter = -1;
        comparison_threshold = -1.0;
    }

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
    void DecisionTreeNode::fit(double** parameters, int* labels, size_t count) {
        //printf("Starting fit of tree, %zu parameters sent.\n", count);
        if (calculate_entropy(labels, count) <= 0) {
            //printf("Node is a leaf!\n");
            // the entropy is 0! This node is pure! Find the first label and set it as the default value.
            default_value = labels[0];
            return;
        }
        //printf("Node isn't a leaf.\n");

        // create list of test splits
        auto** test_splits = new double*[parameter_count];
        for (size_t i = 0; i < parameter_count; ++i){
            // create a list for the test splits of this parameter
            test_splits[i] = new double[count]{0};  // TODO: default everything to -inf

            // add all of the copies of this parameter to the array
            for (size_t j = 0; j < count; j++){
                double val = parameters[j][i];

                // insertion sort it into the array.
                for (size_t k = 0; k < count; k++){
                    if (val > test_splits[i][k]){
                        double temp = val;
                        val = test_splits[i][k];
                        test_splits[i][k] = temp;
                    }
                }
            }

            // finalize the array values
            for (size_t j = 0; j < count - 1; ++j){
                // going to try to use the midpoints between values, not the values themselves.
                test_splits[i][j] = (test_splits[i][j] + test_splits[i][j + 1]) / 2;
            }
        }

        // okay, we've picked our test parameters, now to evaluate.
        size_t best_parameter = 0;
        size_t best_split = 0;
        double best_information_gain_ratio = calculate_information_gain(parameters, labels, count, 0, test_splits[0][0]);
        for (size_t i = 0; i < parameter_count; ++i){
            for (size_t j = 0; j < count - 1; ++j){
                double this_information_gain_ratio = calculate_information_gain(parameters, labels, count, i, test_splits[i][j]);
                //printf("%zu, %zu, %lf, %lf\n", i, j, test_splits[i][j], this_information_gain_ratio);
                if (this_information_gain_ratio > best_information_gain_ratio){
                    best_information_gain_ratio = this_information_gain_ratio;
                    best_parameter = i;
                    best_split = j;
                }
            }
        }

        comparison_parameter = best_parameter;
        comparison_threshold = test_splits[best_parameter][best_split];

        // cleanup. instead of waiting until the very end, we'll clean up now before recursing.
        for (size_t i = 0; i < parameter_count; ++i){
            delete[] test_splits[i];
        }
        delete[] test_splits;

        size_t lesser_count = 0;
        for (size_t i = 0; i < count; i++){
            if (parameters[i][comparison_parameter] < comparison_threshold) ++lesser_count;
        }
        size_t greater_count = count - lesser_count;

        // recursively fit the lesser child
        //printf("Fitting lesser child.\n");
        auto** lesser_parameters = new double*[lesser_count];
        auto* lesser_labels = new int[lesser_count];
        size_t lesser_index = 0;
        for (size_t parent_index = 0; parent_index < count; parent_index++){
            if (parameters[parent_index][comparison_parameter] >= comparison_threshold) continue;
            lesser_parameters[lesser_index] = parameters[parent_index];
            lesser_labels[lesser_index++] = labels[parent_index];
        }
        lesser_branch = new DecisionTreeNode(parameter_count, label_count);
        lesser_branch->fit(lesser_parameters, lesser_labels, lesser_count);
        //printf("Done fitting lesser child.\n");
        delete[] lesser_parameters;
        delete[] lesser_labels;

        // recursively fit the greater child
        //printf("Fitting greater child.\n");
        auto** greater_parameters = new double*[greater_count];
        auto* greater_labels = new int[greater_count];
        size_t greater_index = 0;
        for (size_t parent_index = 0; parent_index < count; parent_index++){
            if (parameters[parent_index][comparison_parameter] < comparison_threshold) continue;
            greater_parameters[greater_index] = parameters[parent_index];
            greater_labels[greater_index++] = labels[parent_index];
        }
        greater_branch = new DecisionTreeNode(parameter_count, label_count);
        greater_branch->fit(greater_parameters, greater_labels, greater_count);
        //printf("Done fitting greater child.\n");
        delete[] greater_parameters;
        delete[] greater_labels;

    }
#pragma clang diagnostic pop

    int DecisionTreeNode::predict_simple(double* parameters) {
        if (lesser_branch == nullptr || greater_branch == nullptr) return default_value;
        if (parameters[comparison_parameter] < comparison_threshold) return lesser_branch->predict(parameters);
        else return greater_branch->predict(parameters);
    }

    [[maybe_unused]] double DecisionTreeNode::calculate_entropy(const int *labels, size_t total_count) const {

        // count how many of each label there are.
        auto* label_counts = new size_t[label_count]{0};
        for (size_t i = 0; i < total_count; i++){
            int label = labels[i];
            ++label_counts[label];
        }

        // calculate the entropy
        double entropy = 0.0;
        for (int i = 0; i < label_count; ++i){  // run the summation
            double p_i = (double)label_counts[i] / (double)total_count;
            if (p_i == 0) continue;
            entropy += p_i * log2(p_i);
        }
        entropy = -entropy;

        delete[] label_counts;
        return entropy;
    }

    [[maybe_unused]] double DecisionTreeNode::calculate_information_gain(double** parameters, const int* labels, size_t total_count, size_t split_parameter, double split_threshold) const {
        // count how many of each label there are.
        auto* parent_label_counts = new size_t[label_count]{0};
        auto* lesser_child_label_counts = new size_t[label_count]{0};
        auto* greater_child_label_counts = new size_t[label_count]{0};
        double lesser_child_split_count = 0;
        double greater_child_split_count = 0;
        for (size_t i = 0; i < total_count; i++){

            //add to the total label count (parent entropy)
            int label = labels[i];
            ++parent_label_counts[label];

            // add to the child label counts
            if (parameters[i][split_parameter] < split_threshold) {
                ++lesser_child_label_counts[label];
                ++lesser_child_split_count;
            }
            else {
                ++greater_child_label_counts[label];
                ++greater_child_split_count;
            }
        }

        // calculate the parent entropy
        double parent_entropy = 0.0;
        for (int i = 0; i < label_count; ++i){  // run the summation
            double p_i = (double)parent_label_counts[i] / (double)total_count;
            if (p_i == 0) continue;
            parent_entropy += p_i * log2(p_i);
        }
        parent_entropy = -parent_entropy;

        //calculate the child entropies
        double lesser_child_entropy = 0.0;
        for (int i = 0; i < label_count; ++i){  // run the summation
            double p_i = (double)lesser_child_label_counts[i] / (double)lesser_child_split_count;
            if (p_i == 0) continue;
            lesser_child_entropy += p_i * log2(p_i);
        }
        lesser_child_entropy = -lesser_child_entropy;
        double greater_child_entropy = 0.0;
        for (int i = 0; i < label_count; ++i){  // run the summation
            double p_i = (double)greater_child_label_counts[i] / (double)greater_child_split_count;
            if (p_i == 0) continue;
            greater_child_entropy += p_i * log2(p_i);
        }
        greater_child_entropy = -greater_child_entropy;

        delete[] parent_label_counts;
        delete[] lesser_child_label_counts;
        delete[] greater_child_label_counts;
        return parent_entropy - ((lesser_child_entropy + greater_child_entropy) / 2);
    }

    [[maybe_unused]] double DecisionTreeNode::calculate_split_information(double** parameters, size_t total_count, size_t split_parameter, double split_threshold) {
        // count how many of each label there are.
        double lesser_child_split_count = 0;
        double greater_child_split_count = 0;
        for (size_t i = 0; i < total_count; i++){

            // add to the child label counts
            if (parameters[i][split_parameter] < split_threshold) ++lesser_child_split_count;
            else ++greater_child_split_count;
        }

        // calculate the split information
        double split_information = -((((double)lesser_child_split_count / (double)total_count) * log2((double)lesser_child_split_count / (double)total_count)) + (((double)greater_child_split_count / (double)total_count) * log2((double)greater_child_split_count / (double)total_count)));

        return split_information;
    }

    [[maybe_unused]] double DecisionTreeNode::calculate_information_gain_ratio(double** parameters, const int* labels, size_t total_count, size_t split_parameter, double split_threshold) const {
        // count how many of each label there are.
        auto* parent_label_counts = new size_t[label_count]{0};
        auto* lesser_child_label_counts = new size_t[label_count]{0};
        auto* greater_child_label_counts = new size_t[label_count]{0};
        double lesser_child_split_count = 0;
        double greater_child_split_count = 0;
        for (size_t i = 0; i < total_count; i++){

            //add to the total label count (parent entropy)
            int label = labels[i];
            ++parent_label_counts[label];

            // add to the child label counts
            if (parameters[i][split_parameter] < split_threshold) {
                ++lesser_child_label_counts[label];
                ++lesser_child_split_count;
            }
            else {
                ++greater_child_label_counts[label];
                ++greater_child_split_count;
            }
        }

        // calculate the parent entropy
        double parent_entropy = 0.0;
        for (int i = 0; i < label_count; ++i){  // run the summation
            double p_i = (double)parent_label_counts[i] / (double)total_count;
            if (p_i == 0) continue;
            parent_entropy += p_i * log2(p_i);
        }
        parent_entropy = -parent_entropy;

        //calculate the child entropies
        double lesser_child_entropy = 0.0;
        for (int i = 0; i < label_count; ++i){  // run the summation
            double p_i = (double)lesser_child_label_counts[i] / (double)lesser_child_split_count;
            if (p_i == 0) continue;
            lesser_child_entropy += p_i * log2(p_i);
        }
        lesser_child_entropy = -lesser_child_entropy;
        double greater_child_entropy = 0.0;
        for (int i = 0; i < label_count; ++i){  // run the summation
            double p_i = (double)greater_child_label_counts[i] / (double)greater_child_split_count;
            if (p_i == 0) continue;
            greater_child_entropy += p_i * log2(p_i);
        }
        greater_child_entropy = -greater_child_entropy;

        // calculate information gain
        double information_gain = parent_entropy - ((lesser_child_entropy + greater_child_entropy) / 2);

        // calculate the split information
        double greater_child_split_info = greater_child_split_count == 0 ? 0 : (((double)greater_child_split_count / (double)total_count) * log2((double)greater_child_split_count / (double)total_count));
        double lesser_child_split_info = lesser_child_split_count == 0 ? 0 : (((double)lesser_child_split_count / (double)total_count) * log2((double)lesser_child_split_count / (double)total_count));
        double split_information = -(lesser_child_split_info + greater_child_split_info);

        delete[] parent_label_counts;
        delete[] lesser_child_label_counts;
        delete[] greater_child_label_counts;
        return split_information == 0 ? 0 :information_gain / split_information;
    }

    int DecisionTreeNode::predict(const double* parameters){
        if (lesser_branch == nullptr || greater_branch == nullptr) return default_value;
        DecisionTreeNode* dtn;
        if (parameters[comparison_parameter] < comparison_threshold) dtn = lesser_branch;
        else dtn = greater_branch;

        while (true){
            if (dtn->lesser_branch == nullptr || dtn->greater_branch == nullptr) return dtn->default_value;
            if (parameters[dtn->comparison_parameter] < dtn->comparison_threshold) dtn = dtn->lesser_branch;
            else dtn = dtn->greater_branch;
        }
    }

} // pico_dt