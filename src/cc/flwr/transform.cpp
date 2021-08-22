// Copyright 2020-present pytorch-cpp Authors
#include "transform.h"
#include <torch/torch.h>

using torch::indexing::Ellipsis;
using torch::indexing::Slice;

namespace transform
{
	namespace
	{
		double rand_double()
		{
			return torch::rand(1)[0].item<double>();
		}

		int64_t rand_int(int64_t max)
		{
			return torch::randint(max, 1)[0].item<int64_t>();
		}
	} // namespace

	// RandomHorizontalFlip
	RandomHorizontalFlip::RandomHorizontalFlip(double p) : p_(p) {}

	torch::Tensor RandomHorizontalFlip::operator()(torch::Tensor input)
	{
		if (rand_double() < p_)
		{
			return input.flip(-1);
		}

		return input;
	}

	// ConstantPad
	ConstantPad::ConstantPad(const std::vector<int64_t> &padding, torch::Scalar value)
		: padding_(padding), value_(value) {}

	ConstantPad::ConstantPad(int64_t padding, torch::Scalar value)
		: padding_(4, padding), value_(value) {}

	torch::Tensor ConstantPad::operator()(torch::Tensor input)
	{
		return torch::constant_pad_nd(input, padding_, value_);
	}

	// RandomCrop
	RandomCrop::RandomCrop(const std::vector<int64_t> &size) : size_(size) {}

	torch::Tensor RandomCrop::operator()(torch::Tensor input)
	{
		auto height_offset_length = input.size(-2) - size_[0];
		auto width_offset_length = input.size(-1) - size_[1];

		auto height_offset = rand_int(height_offset_length);
		auto width_offset = rand_int(width_offset_length);

		return input.index({Ellipsis,
							Slice(height_offset, height_offset + size_[0]), Slice(width_offset, width_offset + size_[1])});
	}
} // namespace transform
