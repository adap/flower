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

class DeepseekV3Attention: Module {
  var config: DeepseekV3Configuration
  var hiddenSize: Int
  var numHeads: Int
  var maxPositionEmbeddings: Int
  var ropeTheta: Float
  var qLoraRank: Int?
  var qkRopeHeadDim: Int
  var kvLoraRank: Int
  var vHeadDim: Int
  var qkNopeHeadDim: Int
  var qHeadDim: Int
  var scale: Float

  var rope: DeepseekV3YarnRotaryEmbedding
  var qProj: Linear?
  var qAProj: Linear?
  var qALayerNorm: RMSNorm?
  var qBProj: Linear?
  var oProj: Linear
  var kvAProjWithMqa: Linear
  var kvALayerNorm: RMSNorm
  var kvBProj: Linear

  init(config: DeepseekV3Configuration) {
    self.config = config
    self.hiddenSize = config.hiddenSize
    self.numHeads = config.numAttentionHeads
    self.maxPositionEmbeddings = config.maxPositionEmbeddings
    self.ropeTheta = config.ropeTheta
    self.qLoraRank = config.qLoraRank
    self.qkRopeHeadDim = config.qkRopeHeadDim
    self.kvLoraRank = config.kvLoraRank
    self.vHeadDim = config.vHeadDim
    self.qkNopeHeadDim = config.qkNopeHeadDim
    self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim

    self.scale = pow(Float(qHeadDim), -0.5)

    if let qLoraRank = qLoraRank {
      self.qAProj = Linear(
        hiddenSize, qLoraRank, bias: config.attentionBias
      )
      self.qALayerNorm = RMSNorm(dimensions: qLoraRank)
      self.qBProj = Linear(
        qLoraRank, numHeads * qHeadDim, bias: false
      )
    } else {
      self.qProj = Linear(hiddenSize, numHeads * qHeadDim, bias: false)
    }

    self.kvAProjWithMqa = Linear(
      hiddenSize,
      kvLoraRank + qkRopeHeadDim,
      bias: config.attentionBias
    )
    self.kvALayerNorm = RMSNorm(dimensions: kvLoraRank)
    self.kvBProj = Linear(
      kvLoraRank,
      numHeads * (qHeadDim - qkRopeHeadDim + vHeadDim),
      bias: false
    )
    self.oProj = Linear(numHeads * vHeadDim, hiddenSize, bias: config.attentionBias)

    guard let ropeScaling = config.ropeScaling,
      case .float(let scalingFactor) = ropeScaling["factor"],
      case .int(let originalMaxPositionEmbeddings) = ropeScaling["original_max_position_embeddings"]
        ?? .int(4096),
      case .float(let betaFast) = ropeScaling["beta_fast"] ?? .float(32),
      case .float(let betaSlow) = ropeScaling["beta_slow"] ?? .float(1),
      case .float(var mscale) = ropeScaling["mscale"] ?? .float(1),
      case .float(let mscaleAllDim) = ropeScaling["mscale_all_dim"] ?? .float(0)
    else {
      self.rope = DeepseekV3YarnRotaryEmbedding(dim: qkRopeHeadDim, base: ropeTheta)
      return
    }
    if mscaleAllDim != 0 {
      mscale = yarnGetMScale(scale: scalingFactor, mscale: mscaleAllDim)
      self.scale = self.scale * mscale * mscale
    }

    self.rope = DeepseekV3YarnRotaryEmbedding(
      dim: qkRopeHeadDim, maxPositionEmbeddings: maxPositionEmbeddings,
      base: ropeTheta,
      scalingFactor: scalingFactor,
      originalMaxPositionEmbeddings: originalMaxPositionEmbeddings,
      betaFast: betaFast,
      betaSlow: betaSlow,
      mscale: mscale,
      mscaleAllDim: mscaleAllDim)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
    let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))

    var q: MLXArray
    if qLoraRank == nil {
      q = self.qProj!(x)
    } else {
      q = self.qBProj!(self.qALayerNorm!(self.qAProj!(x)))
    }

    q = q.reshaped(B, L, self.numHeads, self.qHeadDim).transposed(0, 2, 1, 3)
    let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
    var (qNope, qPe) = (splitQ[0], splitQ[1])
    var compressedKv = self.kvAProjWithMqa(x)
    let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
    compressedKv = splitCompressedKv[0]
    var kPe = splitCompressedKv[1]
    kPe = kPe.reshaped(B, L, 1, self.qkRopeHeadDim).transposed(0, 2, 1, 3)
    var kv = self.kvBProj(kvALayerNorm(compressedKv))
    kv = kv.reshaped(B, L, self.numHeads, -1).transposed(0, 2, 1, 3)
    let splitKv = split(kv, indices: [self.qkNopeHeadDim], axis: -1)

    var (kNope, values) = (splitKv[0], splitKv[1])

    var keys: MLXArray
    if let cache = cache {
      qPe = self.rope(qPe, offset: cache.offset)
      kPe = self.rope(kPe, offset: cache.offset)
      kPe = repeated(kPe, count: numHeads, axis: 1)
      (keys, values) = cache.update(keys: concatenated([kNope, kPe], axis: -1), values: values)
    } else {
      qPe = self.rope(qPe)
      kPe = self.rope(kPe)
      kPe = repeated(kPe, count: numHeads, axis: 1)
      keys = concatenated([kNope, kPe], axis: -1)
    }

    let queries = concatenated([qNope, qPe], axis: -1)

    let output = scaledDotProductAttention(
      queries: queries, keys: keys, values: values, scale: scale, mask: mask
    )
    .transposed(0, 2, 1, 3)
    .reshaped(B, L, -1)

    return self.oProj(output)
  }
}

class DeepseekV3MLP: Module, UnaryLayer {
  var config: DeepseekV3Configuration
  var hiddenSize: Int
  var intermediateSize: Int
  var gateProj: Linear
  var upProj: Linear
  var downProj: Linear

  init(config: DeepseekV3Configuration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
    self.config = config
    self.hiddenSize = hiddenSize ?? config.hiddenSize
    self.intermediateSize = intermediateSize ?? config.intermediateSize
    self.gateProj = Linear(self.hiddenSize, self.intermediateSize, bias: false)
    self.upProj = Linear(self.hiddenSize, self.intermediateSize, bias: false)
    self.downProj = Linear(self.intermediateSize, self.hiddenSize, bias: false)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    self.downProj(silu(self.gateProj(x)) * self.upProj(x))
  }
}

//func groupExpertSelect(
//  gates: MLXArray,
//  eScoreCorrectionBias: MLXArray,
//  topK: Int,
//  nGroup: Int,
//  topkGroup: Int,
//  routedScalingFactor: Double,
//  normTopkProb: Bool
//) -> ([Int], [Double]) {
//  var k = topK
//  var scores = sigmoid(gates)
//  scores = scores + eScoreCorrectionBias
//  
//  let newShape: [Int] = scores.shape.dropLast() + [nGroup, -1]
//  scores = scores.reshaped(newShape)
//  
//  sorted(scores, axis: -1)
////  def group_expert_select(
////      gates,
////      e_score_correction_bias,
////      top_k,
////      n_group,
////      topk_group,
////      routed_scaling_factor,
////      norm_topk_prob,
////  ):
////
////      k = top_k
////      scores = mx.sigmoid(gates.astype(mx.float32))
////      scores = scores + e_score_correction_bias
////      scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
////      group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
////      k = n_group - topk_group
////      group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
////      scores = mx.put_along_axis(scores, group_idx, mx.array(0.0), axis=-2)
////      scores = mx.flatten(scores, -2, -1)
////
////      k = top_k
////      inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
////      scores = mx.take_along_axis(scores, inds, axis=-1)
////      if top_k > 1 and norm_topk_prob:
////          denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
////          scores = scores / denominator
////      scores = scores * routed_scaling_factor
////
////      return inds, scores
//
////  var scores = gates.map { sigmoid($0) }
////  scores = scores.map { $0 + eScoreCorrectionBias }
////
////  let groupSize = scores.count / nGroup
////  var scoresMatrix = (0..<nGroup).map { Array(scores[$0 * groupSize..<$0 * groupSize + groupSize]) }
////
////  var groupScores = scoresMatrix.map { $0.sorted(by: >).prefix(2).reduce(0, +) }
////
////  let k = nGroup - topkGroup
////  let groupIdx = groupScores.enumerated().sorted { $0.element < $1.element }.prefix(k).map {
////    $0.offset
////  }
////
////  for idx in groupIdx {
////    scoresMatrix[idx] = Array(repeating: 0.0, count: scoresMatrix[idx].count)
////  }
////
////  scores = scoresMatrix.flatMap { $0 }
////
////  let topKIndices = (0..<scores.count).sorted { scores[$0] > scores[$1] }.prefix(topK)
////  var topKScores = topKIndices.map { scores[$0] }
////
////  if topK > 1 && normTopkProb {
////    let sumScores = topKScores.reduce(0, +) + 1e-20
////    topKScores = topKScores.map { $0 / sumScores }
////  }
////
////  topKScores = topKScores.map { $0 * routedScalingFactor }
////
////  return (topKIndices, topKScores)
////}
////
////class MoEGate {
////  var config: DeepseekV3Configuration
////  var topK: Int?
////  var normTopkProb: Bool
////  var nRoutedExperts: Int?
////  var routedScalingFactor: Float
////  var nGroup: Int?
////  var topkGroup: Int?
////  var weight: [[Double]]
////  var eScoreCorrectionBias: [Double]
////
////  init(config: DeepseekV3Configuration) {
////    self.config = config
////    self.topK = config.numExpertsPerTok
////    self.normTopkProb = config.normTopkProb
////    self.nRoutedExperts = config.nRoutedExperts
////    self.routedScalingFactor = config.routedScalingFactor
////    self.nGroup = config.nGroup
////    self.topkGroup = config.topkGroup
////
////    guard config.topkMethod == "noaux_tc" else {
////      fatalError("Unsupported topk method.")
////    }
////
////    self.weight = Array(repeating: Array(repeating: 0.0, count: config.hiddenSize), count: self.nRoutedExperts ?? 0)
////    self.eScoreCorrectionBias = Array(repeating: 0.0, count: self.nRoutedExperts ?? 0)
////  }
////
////  func call(_ x: [[Double]]) -> ([Int], [Double]) {
////    let transposedWeight = transpose(weight)
////    let multiplied = matrixMultiply(x, transposedWeight)
////
////    return groupExpertSelect(
////      gates: multiplied.flatMap { $0 },
////      eScoreCorrectionBias: eScoreCorrectionBias.reduce(0, +),
////      topK: topK ?? 1,
////      nGroup: nGroup ?? 1,
////      topkGroup: topkGroup ?? 1,
////      routedScalingFactor: routedScalingFactor,
////      normTopkProb: normTopkProb
////    )
////  }
//}
