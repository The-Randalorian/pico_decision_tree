#include <iostream>
#include "DecisionTreeNode.h"

int main() {
    double* sample_parameters[] = {
            new double[]{1.0},
            new double[]{2.0},
            new double[]{4.0},
            new double[]{5.0},
            new double[]{7.0},
            new double[]{8.0},
    };

    int sample_labels[6] = { 0, 0, 1, 1, 2, 2};

    auto dt_root = pico_dt::DecisionTreeNode(1, 3);

    dt_root.fit(sample_parameters, sample_labels, 6);

    printf("\n===============================\n  Running sample cases.\n===============================\n\n");

    for (auto & sample_parameter : sample_parameters){
        printf("dt(%lf)=%i\n", sample_parameter[0], dt_root.predict(sample_parameter));
    }

    printf("\n===============================\n  Running test cases.\n===============================\n\n");

    printf("dt(%lf)=%i\n", 9.0, dt_root.predict(new double[]{9.0}));
    printf("dt(%lf)=%i\n", 10.0, dt_root.predict(new double[]{10.0}));
    printf("dt(%lf)=%i\n", 0.0, dt_root.predict(new double[]{0.0}));
    printf("dt(%lf)=%i\n", 4.5, dt_root.predict(new double[]{4.5}));
    printf("dt(%lf)=%i\n", 3.0, dt_root.predict(new double[]{3.0}));
    printf("dt(%lf)=%i\n", 2.9, dt_root.predict(new double[]{2.9}));
    printf("dt(%lf)=%i\n", 3.1, dt_root.predict(new double[]{3.1}));

    return 0;
}
