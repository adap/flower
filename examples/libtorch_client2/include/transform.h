/**
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
**/
// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <random>
#include <vector>

namespace transform
{
	class RandomHorizontalFlip : public torch::data::transforms::TensorTransform<torch::Tensor>
	{
	public:
		// Creates a transformation that randomly horizontally flips a tensor.
		//
		// The parameter `p` determines the probability that a tensor is flipped (default = 0.5).
		explicit RandomHorizontalFlip(double p = 0.5);

		torch::Tensor operator()(torch::Tensor input) override;

	private:
		double p_;
	};

	class ConstantPad : public torch::data::transforms::TensorTransform<torch::Tensor>
	{
	public:
		// Creates a transformation that pads a tensor.
		//
		// `padding` is expected to be a vector of size 4 whose entries correspond to the
		// padding of the sides, i.e {left, right, top, bottom}. `value` determines the value
		// for the padded pixels.
		explicit ConstantPad(const std::vector<int64_t> &padding, torch::Scalar value = 0);

		// Creates a transformation that pads a tensor.
		//
		// The padding will be performed using the size `padding` for all 4 sides.
		// `value` determines the value for the padded pixels.
		explicit ConstantPad(int64_t padding, torch::Scalar value = 0);

		torch::Tensor operator()(torch::Tensor input) override;

	private:
		std::vector<int64_t> padding_;
		torch::Scalar value_;
	};

	class RandomCrop : public torch::data::transforms::TensorTransform<torch::Tensor>
	{
	public:
		// Creates a transformation that randomly crops a tensor.
		//
		// The parameter `size` is expected to be a vector of size 2
		// and determines the output size {height, width}.
		explicit RandomCrop(const std::vector<int64_t> &size);
		torch::Tensor operator()(torch::Tensor input) override;

	private:
		std::vector<int64_t> size_;
	};
} // namespace transform
