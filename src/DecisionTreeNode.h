//
// Created by rando on 1/29/24.
//

#ifndef PICO_DT_DECISIONTREENODE_H
#define PICO_DT_DECISIONTREENODE_H

#include <cstddef>

namespace pico_dt {

    class DecisionTreeNode {
    public:

        DecisionTreeNode(size_t p_parameter_count, int p_label_count);

        void fit(double** parameters, int* labels, size_t count);

        int predict_simple(double* parameters);

        int predict(const double* parameters);

        [[maybe_unused]] double calculate_entropy(const int *labels, size_t count) const;

        [[maybe_unused]] double
        calculate_information_gain(double **parameters, const int *labels, size_t count,
                                   size_t split_parameter,
                                   double split_threshold) const;

        [[maybe_unused]] static double calculate_split_information(double **parameters, size_t count, size_t split_parameter,
                                           double split_threshold) ;

        [[maybe_unused]] double calculate_information_gain_ratio(double **parameters, const int *labels, size_t count,
                                                size_t split_parameter, double split_threshold) const;
    private:
        size_t parameter_count;

        int label_count;

        int default_value;

        size_t comparison_parameter;

        double comparison_threshold;

        DecisionTreeNode* lesser_branch;

        DecisionTreeNode* greater_branch;

    };

} // pico_dt

#endif //PICO_DT_DECISIONTREENODE_H
