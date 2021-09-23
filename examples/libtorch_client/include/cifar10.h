// Copyright 2020-present pytorch-cpp Authors
/* 
MIT License

Copyright (c) 2020-present, pytorch-cpp Authors
Copyright (c) 2019-2020 Omkar Prabhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <cstddef>
#include <fstream>
#include <string>

// CIFAR10 dataset
// based on: https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/datasets/mnist.h.
class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10>
{
public:
	// The mode in which the dataset is loaded
	enum Mode
	{
		kTrain,
		kTest
	};

	// Loads the CIFAR10 dataset from the `root` path.
	//
	// The supplied `root` path should contain the *content* of the unzipped
	// CIFAR10 dataset (binary version), available from http://www.cs.toronto.edu/~kriz/cifar.html.
	explicit CIFAR10(const std::string &root, Mode mode = Mode::kTrain);

	// Returns the `Example` at the given `index`.
	torch::data::Example<> get(size_t index) override;

	// Returns the size of the dataset.
	torch::optional<size_t> size() const override;

	// Returns true if this is the training subset of CIFAR10.
	bool is_train() const noexcept;

	// Returns all images stacked into a single tensor.
	const torch::Tensor &images() const;

	// Returns all targets stacked into a single tensor.
	const torch::Tensor &targets() const;

private:
	torch::Tensor images_;
	torch::Tensor targets_;
	Mode mode_;
};