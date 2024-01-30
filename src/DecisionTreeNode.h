//
// Created by rando on 1/29/24.
//

#ifndef PICO_DT_DECISIONTREENODE_H
#define PICO_DT_DECISIONTREENODE_H

#include <cstddef>
#include <cstdint>

#define PICO_DT_LEAF_FLAG 0xAA
#define PICO_DT_BRANCH_FLAG 0xBB

namespace pico_dt {

    class DecisionTreeNode {
    public:
        /// Create a new Decision Tree Node.
        /// \param p_parameter_count The number of parameters this decision tree node can handle.
        /// \param p_label_count The number of labels the decision tree might classify an item as.
        DecisionTreeNode(size_t p_parameter_count, int p_label_count);

        /// Create a new Decision Tree Node. This is mainly for internal use deserializing.
        /// \param p_parameter_count The number of parameters this decision tree node can handle.
        /// \param p_label_count The number of labels the decision tree might classify an item as.
        /// \param p_default_value The default value to use for this node.
        DecisionTreeNode(size_t p_parameter_count, int p_label_count, int p_default_value);

        /// Create a new Decision Tree Node. This is mainly for internal use deserializing.
        /// \param p_parameter_count The number of parameters the decision tree can handle.
        /// \param p_label_count The number of labels the decision tree might classify an item as.
        /// \param p_comparison_parameter The parameter this node should compare by.
        /// \param p_comparison_threshold The threshold this node should compare against.
        /// \param p_lesser_branch The node for parameters lesser than the test threshold.
        /// \param p_greater_branch The node for parameters greater than the test threshold.
        DecisionTreeNode(size_t p_parameter_count, int p_label_count, size_t p_comparison_parameter,
                         double p_comparison_threshold, DecisionTreeNode *p_lesser_branch,
                         DecisionTreeNode *p_greater_branch);

        /// Fit a decision tree to a given set of parameters and labels. This is called recursively.
        /// \param parameters An array of pointers pointing to arrays of parameters. Arrays of parameters must be parameter_count long.
        /// \param labels An array of labels, with one label for each parameter array given.
        /// \param count The length of both the parameter pointer array (parameters) and label array (labels).
        void fit(double **parameters, int *labels, size_t count);

        /// Simplest prediction method, based on recursion. Recommended not to use.
        /// \param parameters An array of parameters to use.
        /// \return The predicted value.
        [[maybe_unused]] int predict_simple(double *parameters);

        /// Predict a value given some parameters.
        /// \param parameters An array of parameters to use.
        /// \return The predicted valeue.
        [[maybe_unused]] int predict(const double *parameters);

        /// Calculate the entropy at this node. Mainly used internally. See https://en.wikipedia.org/wiki/Entropy_(information_theory)
        /// \param labels An array of labels present at this node.
        /// \param count The number of labels given.
        /// \return The entropy of this node.
        double calculate_entropy(const int *labels, size_t count) const;

        /// Calculate the information gain at this node. Mainly used internally. See https://en.wikipedia.org/wiki/Information_gain_(decision_tree)
        /// \param parameters An array of pointers to arrays of doubles giving sets of input parameters.
        /// \param labels An array of labels present at this node.
        /// \param count The number of labels and parameters given.
        /// \param split_parameter Which parameter to split the data on.
        /// \param split_threshold What value to split the data at.
        /// \return The information gain of this node when split by the threshold.
        [[maybe_unused]] double
        calculate_information_gain(double **parameters, const int *labels, size_t count,
                                   size_t split_parameter,
                                   double split_threshold) const;

        /// Calculate the information gain at this node. Mainly used internally. See https://en.wikipedia.org/wiki/Information_gain_ratio#Split_Information_calculation
        /// \param parameters An array of pointers to arrays of doubles giving sets of input parameters.
        /// \param count The number of parameters given.
        /// \param split_parameter Which parameter to split the data on.
        /// \param split_threshold What value to split the data at.
        /// \return The split information of this node when split by the threshold.
        [[maybe_unused]] static double
        calculate_split_information(double **parameters, size_t count, size_t split_parameter,
                                    double split_threshold);

        /// Calculate the information gain ratio at this node. Mainly used internally. See https://en.wikipedia.org/wiki/Information_gain_ratio
        /// \param parameters An array of pointers to arrays of doubles giving sets of input parameters.
        /// \param labels An array of labels present at this node.
        /// \param count The number of labels and parameters given.
        /// \param split_parameter Which parameter to split the data on.
        /// \param split_threshold What value to split the data at.
        /// \return The information gain of this node when split by the threshold.
        [[maybe_unused]] double calculate_information_gain_ratio(double **parameters, const int *labels, size_t count,
                                                                 size_t split_parameter, double split_threshold) const;
        /// Calculate how large this decision tree will be once serialized.
        /// \return The final size of the serialized decision tree.
        size_t calculate_serialized_size();

        /// Serialize the decision tree into raw bytes.
        /// \return A pointer to a buffer containing the serialized decision tree.
        uint8_t *serialize();

    private:
        size_t parameter_count;

        int label_count;

        int default_value;

        size_t comparison_parameter;

        double comparison_threshold;

        DecisionTreeNode *lesser_branch;

        DecisionTreeNode *greater_branch;

        DecisionTreeNode *parent_branch;

        void serialize_leaf(uint8_t *location);

        void serialize_branch(uint8_t *location);
    };

    /// Create a new decision tree from serialized data.
    /// \param parameter_count How many input parameters the tree will accept.
    /// \param label_count How many labels the tree will group samples into.
    /// \param buffer pointer to the serialized tree data.
    /// \param buffer_length length of the serialized data buffer.
    /// \return A pointer to a new decision tree, made from the serialized data.
    DecisionTreeNode *
    deserialize_decision_tree(size_t parameter_count, int label_count, uint8_t *buffer, size_t buffer_length);

} // pico_dt

#endif //PICO_DT_DECISIONTREENODE_H
