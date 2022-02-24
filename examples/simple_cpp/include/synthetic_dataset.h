//
// Created by Andreea Zaharia on 31/01/2022.
//

#ifndef FLOWER_CPP_SYNTHETIC_DATASET_H
#define FLOWER_CPP_SYNTHETIC_DATASET_H

#include <vector>

class SyntheticDataset {

public:
    // Generates the synthetic dataset of size size around random vector m of size ms_size.
    SyntheticDataset(std::vector<double> ms,double b, size_t training_size, size_t validation_size, size_t  test_size);

    // Returns the size of the dataset.
    size_t size();

    // Returns the training subset of the dataset.
    std::vector<std::vector<double> > get_training_data();

    // Returns the validation subset of the dataset.
    std::vector<std::vector<double> > get_validation_data();

    // Returns the test subset of the dataset.
    std::vector<std::vector<double> > get_test_data();

    int get_features_count();

private:
    std::vector<double> ms;
    double b;
    size_t training_size, validation_size, test_size;

    // The label is the last position in the vector.
    // TODO: consider changing this to a pair with the label.

    std::vector<std::vector<double> > training_data;
    std::vector<std::vector<double> > validation_data;
    std::vector<std::vector<double> > test_data;
};


#endif //FLOWER_CPP_SYNTHETIC_DATASET_H