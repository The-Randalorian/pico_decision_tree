#include <iostream>
#include "DecisionTreeNode.h"

int main() {
    double* sample_parameters[] = {
            new double[]{0, 0, 1.0},
            new double[]{0, 0, 2.0},
            new double[]{0, 0, 4.0},
            new double[]{0, 0, 5.0},
            new double[]{0, 0, 7.0},
            new double[]{0, 0, 8.0},
            new double[]{0, 2, 1.0},
            new double[]{0, 2, 2.0},
            new double[]{0, 2, 4.0},
            new double[]{0, 2, 5.0},
            new double[]{0, 2, 7.0},
            new double[]{0, 2, 8.0},
            new double[]{4, 0, 1.0},
            new double[]{4, 0, 2.0},
            new double[]{4, 0, 4.0},
            new double[]{4, 0, 5.0},
            new double[]{4, 0, 7.0},
            new double[]{4, 0, 8.0},
            new double[]{4, 2, 1.0},
            new double[]{4, 2, 2.0},
            new double[]{4, 2, 4.0},
            new double[]{4, 2, 5.0},
            new double[]{4, 2, 7.0},
            new double[]{4, 2, 8.0},
    };

    int sample_labels[24] = { 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11,11};

    auto dt_root = pico_dt::DecisionTreeNode(3, 12);

    dt_root.fit(sample_parameters, sample_labels, 24);

    printf("\n===============================\n  Running sample cases.\n===============================\n\n");

    for (auto & sample_parameter : sample_parameters){
        printf("dt(%lf, %lf, %lf)=%i\n", sample_parameter[0], sample_parameter[1], sample_parameter[2], dt_root.predict(sample_parameter));
    }

    printf("\n===============================\n  Running test cases.\n===============================\n\n");

    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 9.0, dt_root.predict(new double[]{0, 0, 9.0}));
    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 10.0, dt_root.predict(new double[]{0, 0, 10.0}));
    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 0.0, dt_root.predict(new double[]{0, 0, 0.0}));
    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 4.5, dt_root.predict(new double[]{0, 0, 4.5}));
    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 3.0, dt_root.predict(new double[]{0, 0, 3.0}));
    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 2.9, dt_root.predict(new double[]{0, 0, 2.9}));
    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 3.1, dt_root.predict(new double[]{0, 0, 3.1}));

    printf("\n===============================\n  Testing serialization.\n===============================\n\n");

    uint8_t* copied_buffer = dt_root.serialize();

    for (int i = 0; i < dt_root.calculate_serialized_size(); i++)
    {
        if (i > 0) printf(" ");
        printf("%02X", copied_buffer[i]);
    }
    printf("\n");

    printf("\n===============================\n  Testing deserialization.\n===============================\n\n");

    auto* dt_copy = pico_dt::deserialize_decision_tree(3, 12, copied_buffer, dt_root.calculate_serialized_size());
    printf("Done.\n");

    printf("\n===============================\n  Running sample cases.\n===============================\n\n");

    for (auto & sample_parameter : sample_parameters){
        printf("dt(%lf, %lf, %lf)=%i\n", sample_parameter[0], sample_parameter[1], sample_parameter[2], dt_copy->predict(sample_parameter));
    }

    printf("\n===============================\n  Running test cases.\n===============================\n\n");

    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 9.0, dt_copy->predict(new double[]{0, 0, 9.0}));
    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 10.0, dt_copy->predict(new double[]{0, 0, 10.0}));
    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 0.0, dt_copy->predict(new double[]{0, 0, 0.0}));
    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 4.5, dt_copy->predict(new double[]{0, 0, 4.5}));
    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 3.0, dt_copy->predict(new double[]{0, 0, 3.0}));
    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 2.9, dt_copy->predict(new double[]{0, 0, 2.9}));
    printf("dt(%lf, %lf, %lf)=%i\n", 0.0, 0.0, 3.1, dt_copy->predict(new double[]{0, 0, 3.1}));

    return 0;
}
