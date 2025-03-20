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
import MLXLLM
import MLXLMCommon

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
  numRotations: Double, dim: Double, base: Double = 10000, maxPositionEmbeddings: Double = 2048
) -> Double {
  return (dim * log(maxPositionEmbeddings / (numRotations * 2 * Double.pi))) / (2 * log(base))
}

func yarnFindCorrectionRange(
  lowRot: Double, highRot: Double, dim: Double, base: Double = 10000,
  maxPositionEmbeddings: Double = 2048
) -> (Int, Int) {
  let low = Int(
    floor(
      yarnFindCorrectionDim(
        numRotations: lowRot, dim: dim, base: base, maxPositionEmbeddings: maxPositionEmbeddings)))
  let high = Int(
    ceil(
      yarnFindCorrectionDim(
        numRotations: highRot, dim: dim, base: base, maxPositionEmbeddings: maxPositionEmbeddings)))
  return (max(low, 0), min(high, Int(dim - 1)))
}

func yarnGetMScale(scale: Double = 1, mscale: Double = 1) -> Double {
  return scale <= 1 ? 1.0 : 0.1 * mscale * log(scale) + 1.0
}

func yarnLinearRampMask(minVal: Double, maxVal: Double, dim: Int) -> MLXArray {
  let updatedMaxVal = minVal == maxVal ? maxVal + 0.001 : maxVal
  let linearFunc = (MLXArray(0..<dim) - minVal) / (updatedMaxVal - minVal)
  return clip(linearFunc, min: 0, max: 1)
}
