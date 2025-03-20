// Copyright 2025 Flower Labs GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN

struct DeepseekV3Configuration: Codable, Sendable {
  var modelType: String = "deepseek_v3"
  var vocabSize: Int = 102400
  var hiddenSize: Int = 4096
  var intermediateSize: Int = 11008
  var moeIntermediateSize: Int = 1407
  var numHiddenLayers: Int = 30
  var numAttentionHeads: Int = 32
  var numKeyValueHeads: Int = 32
  var nSharedExperts: Int?
  var nRoutedExperts: Int?
  var routedScalingFactor: Float = 1.0
  var kvLoraRank: Int = 512
  var qLoraRank: Int = 1536
  var qkRopeHeadDim: Int = 64
  var vHeadDim: Int = 128
  var qkNopeHeadDim: Int = 128
  var topkMethod: String = "noaux_tc"
  var scoringFunc: String = "sigmoid"
  var normTopkProb: Bool = true
  var nGroup: Int?
  var topkGroup: Int?
  var numExpertsPerTok: Int?
  var moeLayerFreq: Int = 1
  var firstKDenseReplace: Int = 0
  var maxPositionEmbeddings: Int = 2048
  var rmsNormEps: Float = 1e-6
  var ropeTheta: Float = 10000.0
  var ropeScaling: [String: StringOrNumber]? = nil
  var attentionBias: Bool = false

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case vocabSize = "vocab_size"
    case hiddenSize = "hidden_size"
    case intermediateSize = "intermediate_size"
    case moeIntermediateSize = "moe_intermediate_size"
    case numHiddenLayers = "num_hidden_layers"
    case numAttentionHeads = "num_attention_heads"
    case numKeyValueHeads = "num_key_value_heads"
    case nSharedExperts = "n_shared_experts"
    case nRoutedExperts = "n_routed_experts"
    case routedScalingFactor = "routed_scaling_factor"
    case kvLoraRank = "kv_lora_rank"
    case qLoraRank = "q_lora_rank"
    case qkRopeHeadDim = "qk_rope_head_dim"
    case vHeadDim = "v_head_dim"
    case qkNopeHeadDim = "qk_nope_head_dim"
    case topkMethod = "topk_method"
    case scoringFunc = "scoring_func"
    case normTopkProb = "norm_topk_prob"
    case nGroup = "n_group"
    case topkGroup = "topk_group"
    case numExpertsPerTok = "num_experts_per_tok"
    case moeLayerFreq = "moe_layer_freq"
    case firstKDenseReplace = "first_k_dense_replace"
    case maxPositionEmbeddings = "max_position_embeddings"
    case rmsNormEps = "rms_norm_eps"
    case ropeTheta = "rope_theta"
    case ropeScaling = "rope_scaling"
    case attentionBias = "attention_bias"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.modelType = try container.decode(String.self, forKey: .modelType)
    self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
    self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
    self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
    self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
    self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
    self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
    self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
    self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
    self.nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
    self.routedScalingFactor = try container.decode(Float.self, forKey: .routedScalingFactor)
    self.kvLoraRank = try container.decode(Int.self, forKey: .kvLoraRank)
    self.qLoraRank = try container.decode(Int.self, forKey: .qLoraRank)
    self.qkRopeHeadDim = try container.decode(Int.self, forKey: .qkRopeHeadDim)
    self.vHeadDim = try container.decode(Int.self, forKey: .vHeadDim)
    self.qkNopeHeadDim = try container.decode(Int.self, forKey: .qkNopeHeadDim)
    self.topkMethod = try container.decode(String.self, forKey: .topkMethod)
    self.scoringFunc = try container.decode(String.self, forKey: .scoringFunc)
    self.normTopkProb = try container.decode(Bool.self, forKey: .normTopkProb)
    self.nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup)
    self.topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup)
    self.numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
    self.moeLayerFreq = try container.decode(Int.self, forKey: .moeLayerFreq)
    self.firstKDenseReplace = try container.decode(Int.self, forKey: .firstKDenseReplace)
    self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
    self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
    self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
    self.ropeScaling = try container.decodeIfPresent(
      [String: StringOrNumber].self, forKey: .ropeScaling)
    self.attentionBias = try container.decode(Bool.self, forKey: .attentionBias)
  }
}

func yarnFindCorrectionDim(
  numRotations: Float, dim: Float, base: Float = 10000, maxPositionEmbeddings: Float = 2048
) -> Float {
  return (dim * log(maxPositionEmbeddings / (numRotations * 2 * Float.pi))) / (2 * log(base))
}

func yarnFindCorrectionRange(
  lowRot: Float, highRot: Float, dim: Float, base: Float = 10000,
  maxPositionEmbeddings: Float = 2048
) -> (Float, Float) {
  let low = floor(
    yarnFindCorrectionDim(
      numRotations: lowRot, dim: dim, base: base, maxPositionEmbeddings: maxPositionEmbeddings))
  let high = ceil(
    yarnFindCorrectionDim(
      numRotations: highRot, dim: dim, base: base, maxPositionEmbeddings: maxPositionEmbeddings))
  return (max(low, 0), min(high, dim - 1))
}

func yarnGetMScale(scale: Float = 1, mscale: Float = 1) -> Float {
  return scale <= 1 ? 1.0 : 0.1 * mscale * log(scale) + 1.0
}

func yarnLinearRampMask(minVal: Float, maxVal: Float, dim: Int) -> MLXArray {
  let updatedMaxVal = minVal == maxVal ? maxVal + 0.001 : maxVal
  let linearFunc = (MLXArray(0..<dim) - minVal) / (updatedMaxVal - minVal)
  return clip(linearFunc, min: 0, max: 1)
}

class DeepseekV3YarnRotaryEmbedding: Module {
  var mscale: Float
  var freqs: MLXArray
  init(
    dim: Int,
    maxPositionEmbeddings: Int = 2048,
    base: Float = 10000,
    scalingFactor: Float = 1.0,
    originalMaxPositionEmbeddings: Int = 4096,
    betaFast: Float = 32,
    betaSlow: Float = 1,
    mscale: Float = 1,
    mscaleAllDim: Float = 0
  ) {
    self.mscale =
      yarnGetMScale(scale: scalingFactor, mscale: mscale)
      / yarnGetMScale(scale: scalingFactor, mscale: mscaleAllDim)
    let freqExtra = base ** (MLXArray(stride(from: 0, to: dim, by: 2)) / dim)
    let freqInter = scalingFactor * base ** (MLXArray(stride(from: 0, to: dim, by: 2)) / dim)
    let (low, high) = yarnFindCorrectionRange(
      lowRot: betaFast, highRot: betaSlow, dim: Float(dim), base: base,
      maxPositionEmbeddings: Float(originalMaxPositionEmbeddings))

    let freqMask = 1.0 - yarnLinearRampMask(minVal: low, maxVal: high, dim: dim / 2)

    self.freqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
  }

  func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
    MLXFast.RoPE(
      self.mscale != 1.0 ? self.mscale * x : x,
      dimensions: x.shape[-1],
      traditional: true,
      base: nil,
      scale: 1.0,
      offset: offset,
      freqs: freqs
    )
  }
}

func clippedSilu(_ x: MLXArray) -> MLXArray {
  clip(x * sigmoid(x), min: -100, max: 100)
}


