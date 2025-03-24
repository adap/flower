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
  let dim: Int
  let maxPositionEmbeddings: Int
  let base: Float
  let scalingFactor: Float
  let originalMaxPositionEmbeddings: Int
  let betaFast: Float
  let betaSlow: Float
  
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
    self.dim = dim
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.base = base
    self.scalingFactor = scalingFactor
    self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
    self.betaFast = betaFast
    self.betaSlow = betaSlow
  }

  func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
    let freqExtra = base ** (MLXArray(stride(from: 0, to: dim, by: 2)) / dim)
    let freqInter = scalingFactor * base ** (MLXArray(stride(from: 0, to: dim, by: 2)) / dim)
    let (low, high) = yarnFindCorrectionRange(
      lowRot: betaFast, highRot: betaSlow, dim: Float(dim), base: base,
      maxPositionEmbeddings: Float(originalMaxPositionEmbeddings))

    let freqMask = 1.0 - yarnLinearRampMask(minVal: low, maxVal: high, dim: dim / 2)

    let freqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
    return MLXFast.RoPE(
      self.mscale != 1.0 ? self.mscale * x : x,
      dimensions: x.shape.last ?? 0,
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

  let rope: DeepseekV3YarnRotaryEmbedding
  @ModuleInfo(key: "q_proj") var qProj: Linear?
  @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
  @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
  @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
  @ModuleInfo(key: "o_proj") var oProj: Linear
  @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
  @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
  @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear

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
      self._qAProj.wrappedValue = Linear(
        hiddenSize, qLoraRank, bias: config.attentionBias
      )
      self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank)
      self._qBProj.wrappedValue = Linear(
        qLoraRank, numHeads * qHeadDim, bias: false
      )
    } else {
      self._qProj.wrappedValue = Linear(hiddenSize, numHeads * qHeadDim, bias: false)
    }

    self._kvAProjWithMqa.wrappedValue = Linear(
      hiddenSize,
      kvLoraRank + qkRopeHeadDim,
      bias: config.attentionBias
    )
    self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank)
    self._kvBProj.wrappedValue = Linear(
      kvLoraRank,
      numHeads * (qHeadDim - qkRopeHeadDim + vHeadDim),
      bias: false
    )
    self._oProj.wrappedValue = Linear(numHeads * vHeadDim, hiddenSize, bias: config.attentionBias)

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
    let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

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
  @ModuleInfo(key: "gate_proj") var gateProj: Linear
  @ModuleInfo(key: "up_proj") var upProj: Linear
  @ModuleInfo(key: "down_proj") var downProj: Linear

  init(config: DeepseekV3Configuration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
    self.config = config
    self.hiddenSize = hiddenSize ?? config.hiddenSize
    self.intermediateSize = intermediateSize ?? config.intermediateSize
    self._gateProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
    self._upProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
    self._downProj.wrappedValue = Linear(self.intermediateSize, self.hiddenSize, bias: false)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    self.downProj(silu(self.gateProj(x)) * self.upProj(x))
  }
}

class MoEGate: Module {
  var config: DeepseekV3Configuration
  var topK: Int?
  var normTopkProb: Bool
  var nRoutedExperts: Int?
  var routedScalingFactor: Float
  var nGroup: Int
  var topkGroup: Int?
  
  var weight: MLXArray
  var e_score_correction_bias: MLXArray

  init(config: DeepseekV3Configuration) {
    self.config = config
    self.topK = config.numExpertsPerTok
    self.normTopkProb = config.normTopkProb
    self.nRoutedExperts = config.nRoutedExperts
    self.routedScalingFactor = config.routedScalingFactor
    self.nGroup = config.nGroup ?? 1
    self.topkGroup = config.topkGroup

    guard config.topkMethod == "noaux_tc" else {
      fatalError("Unsupported topk method.")
    }

    self.weight = zeros([self.nRoutedExperts ?? 1, config.hiddenSize])
    self.e_score_correction_bias = zeros([self.nRoutedExperts ?? 1])
  }
  
  func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
    print("shape: \(x.shape)")
    let (bsz, seqLen, h) = (x.dim(0), x.dim(1), x.dim(2))
    
    let hiddenStates = x.matmul(weight.T)
    print(hiddenStates.shape)
    var scores = sigmoid(hiddenStates)
    print("scores shape ", scores.shape)
    let scoresForChoice = scores + e_score_correction_bias
    print(scoresForChoice.shape)
    let groupScores = scoresForChoice.reshaped(bsz, seqLen, self.nGroup, -1)
    print(groupScores.shape)
    let topKGroup = sorted(groupScores, axis: -1)[.ellipsis, ..<2].sum(axis: -1, keepDims: true)
    print("group scores topK shape \(topKGroup.shape)")
    var k = nGroup - (topkGroup ?? 1)
    print("k ", k)
    let groupIdx = argPartition(topKGroup, kth: k-1, axis: -2)[.ellipsis, ..<k, 0...]
    print("group idx shape \(topKGroup.shape)")
    scores = putAlong(groupScores, groupIdx, values: MLXArray(0.0), axis: -2)
    scores = flattened(scores, start: -2, end: -1)
    
    k = topK ?? 1
    let inds = argPartition(-scores, kth: k-1, axis: -1)[.ellipsis, ..<k]
    scores = takeAlong(scores, inds, axis: -1)
    if topK ?? 1 > 1, normTopkProb {
      let denominator = scores.sum(axis:-1, keepDims: true) + 1e-20
      scores = scores / denominator
      scores = scores * routedScalingFactor
    }

    return (inds, scores)
  }
}

class DeepseekV3MoE: Module, UnaryLayer {
  var config: DeepseekV3Configuration
  var numExpertsPerTok: Int
  @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
  var gate: MoEGate
  @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV3MLP?

  init(config: DeepseekV3Configuration) {
    self.config = config
    self.numExpertsPerTok = config.numExpertsPerTok ?? 1

    self._switchMLP.wrappedValue = SwitchGLU(
      inputDims: config.hiddenSize,
      hiddenDims: config.moeIntermediateSize,
      numExperts: config.nRoutedExperts ?? 1,
      activation: clippedSilu
    )

    self.gate = MoEGate(config: config)

    if let sharedExpertCount = config.nSharedExperts {
      let intermediateSize = config.moeIntermediateSize * sharedExpertCount
      self._sharedExperts.wrappedValue = DeepseekV3MLP(config: config, intermediateSize: intermediateSize)
    }
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let (indices, scores) = gate(x)
    var y = switchMLP(x, indices)
    y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

    if let shared = sharedExperts {
      y = y + shared(x)
    }
    return y
  }
}

class DeepseekV3DecoderLayer: Module {
  @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV3Attention
  var mlp: UnaryLayer
  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  init(config: DeepseekV3Configuration, layerIdx: Int) {
    self._selfAttn.wrappedValue = DeepseekV3Attention(config: config)

    if config.nRoutedExperts != nil,
       layerIdx >= config.firstKDenseReplace,
       layerIdx % config.moeLayerFreq == 0 {
      self.mlp = DeepseekV3MoE(config: config)
    } else {
      self.mlp = DeepseekV3MLP(config: config)
    }

    self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
    let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
    let h = x + r
    let r2 = mlp(postAttentionLayerNorm(h))
    return h + r2
  }
}

class DeepseekV3ModelInner: Module {
  var config: DeepseekV3Configuration
  var vocabSize: Int
  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  var layers: [DeepseekV3DecoderLayer]
  var startIdx: Int
  var endIdx: Int
  var numLayers: Int
  @ModuleInfo(key: "norm") var norm: RMSNorm
  var pipelineRank: Int
  var pipelineSize: Int

  init(config: DeepseekV3Configuration) {
    self.config = config
    self.vocabSize = config.vocabSize
    self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
    self.layers = (0..<config.numHiddenLayers).map { DeepseekV3DecoderLayer(config: config, layerIdx: $0) }
    self.startIdx = 0
    self.endIdx = layers.count
    self.numLayers = endIdx
    self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    self.pipelineRank = 0
    self.pipelineSize = 1
  }

  func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
    var h = embedTokens(x)
    
    let attentionMask = createAttentionMask(h: h, cache: cache)

    for (i, layer) in layers.enumerated() {
        h = layer(h, mask: attentionMask, cache: cache?[i])
    }

    return norm(h)
  }
}

class DeepseekV3Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
  var kvHeads: [Int] = []
  
  var args: DeepseekV3Configuration
  var modelType: String
  var model: DeepseekV3ModelInner
  @ModuleInfo(key: "lm_head") var lmHead: Linear

  init(_ args: DeepseekV3Configuration) {
    self.args = args
    self.modelType = args.modelType
    self.model = DeepseekV3ModelInner(config: args)
    self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
  }

  func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
    let out = model(inputs, cache: cache)
    return lmHead(out)
  }

  func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    var newWeights = weights

    func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
      let bs = 128
      let (m, n) = (weight.shape[0], weight.shape[1])
      let padBottom = (bs - m % bs) % bs
      let padSide = (bs - n % bs) % bs
      
      var padded = padded(weight, widths:[.init((0, padBottom)), .init((0, padSide))])
      padded = padded.reshaped([ (m + padBottom) / bs, bs, (n + padSide) / bs, bs ])
      let scaled = padded * scaleInv[0..., .newAxis, 0..., .newAxis]
      return scaled.reshaped([m + padBottom, n + padSide])[0..<m, 0..<n]
    }

    for (key, value) in weights {
      if key.contains("weight_scale_inv") {
        let scaleKey = key
        let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
        let dequantized = dequant(weight: weights[weightKey]!, scaleInv: value)
        newWeights[weightKey] = dequantized
      }
    }

    for l in 0..<args.numHiddenLayers {
      let prefix = "model.layers.\(l)"
      for (wName, projName) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
        for key in ["weight", "scales", "biases"] {
          let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
          if weights[firstKey] != nil {
            let joined = (0..<(args.nRoutedExperts ?? 1)).map {
              weights["\(prefix).mlp.experts.\($0).\(projName).\(key)"]!
            }
            newWeights["\(prefix).mlp.switch_mlp.\(projName).\(key)"] = stacked(joined)
          }
        }
      }
    }

    return newWeights.filter { key, _ in
      !key.starts(with: "model.layers.61") && !key.contains("rotary_emb.inv_freq")
    }
  }
  
  public func loraLinearLayers() -> LoRALinearLayers {
    model.layers.map { ($0.selfAttn, ["q_proj", "v_proj"]) }
  }
}
