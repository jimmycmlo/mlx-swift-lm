// Copyright © 2024 Apple Inc.

// port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen2_vl

import AVFoundation
import CoreImage
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Language

private enum Language {

    /// Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors
    static private func applyMultimodalRotaryPositionEmbedding(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray,
        positionIds: MLXArray, mropeSection: [Int]
    ) -> (MLXArray, MLXArray) {
        var cos = cos[positionIds]
        var sin = sin[positionIds]

        cos =
            concatenated(
                // [m[i % 3] for i, m in enumerate(mx.split(cos, mrope_section, axis=-1))]
                split(cos, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        sin =
            concatenated(
                split(sin, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        // Apply rotary embedding
        let qEmbed = (q * cos) + (QwenVL.rotateHalf(q) * sin)
        let kEmbed = (k * cos) + (QwenVL.rotateHalf(k) * sin)
        return (qEmbed, kEmbed)
    }

    fileprivate class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let mropeSection: [Int]

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        public init(_ args: Qwen2VLConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            if let v = args.ropeScaling?["mrope_section"], let array = v.asInts() {
                // mrope_section = np.cumsum(mrope_section * 2)[:-1].tolist()
                self.mropeSection = sequence(state: (0, array.makeIterator())) { state in
                    if let v = state.1.next() {
                        // note the *2
                        state.0 += v * 2
                        return state.0
                    } else {
                        return nil
                    }
                }.dropLast()
            } else {
                fatalError("rope_scaling['mrope_section'] must be an array of integers")
            }

            self._rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            // prepare the queries, keys and values for the attention computation
            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            queries = rotaryEmbedding(queries, offset: offset)
            keys = rotaryEmbedding(keys, offset: offset)

            let maskConverted: MLXFast.ScaledDotProductAttentionMaskMode =
                if let mask {
                    .array(mask[.ellipsis, 0 ..< keys.dim(-2)])
                } else {
                    .none
                }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: maskConverted
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class Qwen2VLDecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: Qwen2VLConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            let out = h + r
            return out
        }
    }

    fileprivate class Qwen2Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [Qwen2VLDecoderLayer]
        fileprivate let norm: RMSNorm

        public init(_ args: Qwen2VLConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.layers = (0 ..< args.hiddenLayers)
                .map { _ in
                    Qwen2VLDecoderLayer(args)
                }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let mask: MLXArray? = createAttentionMask(h: h, cache: cache)

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return norm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Qwen2Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]

        public init(_ args: Qwen2VLConfiguration.TextConfiguration) {
            self.model = Qwen2Model(args)

            if !args.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
            }

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> LMOutput {
            var out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Vision

private enum Vision {

    static fileprivate func applyMultimodalRotaryPositionEmbedding(
        _ tensor: MLXArray, freqs: MLXArray
    ) -> MLXArray {
        var cos = cos(freqs)
        var sin = sin(freqs)

        cos = expandedDimensions(cos, axis: 1)
        cos = tiled(cos, repetitions: [1, 1, 2])
        cos = expandedDimensions(cos, axis: 0)

        sin = expandedDimensions(sin, axis: 1)
        sin = tiled(sin, repetitions: [1, 1, 2])
        sin = expandedDimensions(sin, axis: 0)

        let output = (tensor * cos) + (QwenVL.rotateHalf(tensor) * sin)
        return output.asType(tensor.dtype)
    }

    fileprivate class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        @ModuleInfo(key: "ln_q") var layerNormQ: LayerNorm
        @ModuleInfo var mlp: (Linear, GELU, Linear)

        init(dimensions: Int, contextDimensions: Int, spatialMergeSize: Int) {
            self.hiddenSize = contextDimensions * (spatialMergeSize * spatialMergeSize)
            self._layerNormQ.wrappedValue = LayerNorm(dimensions: contextDimensions, eps: 1e-6)
            self.mlp = (
                Linear(hiddenSize, hiddenSize),
                GELU(),
                Linear(hiddenSize, dimensions)
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = layerNormQ(x).reshaped(-1, hiddenSize)
            x = mlp.0(x)
            x = mlp.1(x)
            x = mlp.2(x)
            return x
        }
    }

    fileprivate class Attention: Module {

        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        public init(dims: Int, numHeads: Int) {
            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(dims, 3 * dims, bias: true)
            self._proj.wrappedValue = Linear(dims, dims)
        }

        public func callAsFunction(
            _ x: MLXArray, frames: [THW], rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)
            let B = frames[0].t
            let L = sequenceLength / B

            let qkv = qkv(x)
            let s = split(qkv, parts: 3, axis: -1)
            var (q, k, v) = (s[0], s[1], s[2])

            q = q.reshaped(sequenceLength, numHeads, -1)
            k = k.reshaped(sequenceLength, numHeads, -1)
            v = v.reshaped(sequenceLength, numHeads, -1)

            q = applyMultimodalRotaryPositionEmbedding(q, freqs: rotaryPositionEmbedding)
            k = applyMultimodalRotaryPositionEmbedding(k, freqs: rotaryPositionEmbedding)

            q = q.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            k = k.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            v = v.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return proj(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo var activation: GELU
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self.activation = GELU(approximation: .fast)
            self.fc1 = Linear(dimensions, hiddenDimensions)
            self.fc2 = Linear(hiddenDimensions, dimensions)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    fileprivate class Qwen2VLVisionBlock: Module {

        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        public init(_ config: Qwen2VLConfiguration.VisionConfiguration) {
            self.norm1 = LayerNorm(dimensions: config.embedDimensions, eps: 1e-6)
            self.norm2 = LayerNorm(dimensions: config.embedDimensions, eps: 1e-6)

            self._attention.wrappedValue = Attention(
                dims: config.embedDimensions, numHeads: config.numHeads)

            let mlpHiddenDimensions = Int(Float(config.embedDimensions) * config.mlpRatio)
            self.mlp = MLP(
                dimensions: config.embedDimensions, hiddenDimensions: mlpHiddenDimensions)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray, frames: [THW], rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            var hiddenStates =
                hiddenStates
                + attention(
                    norm1(hiddenStates),
                    frames: frames,
                    rotaryPositionEmbedding: rotaryPositionEmbedding
                )
            hiddenStates = hiddenStates + mlp(norm2(hiddenStates))
            return hiddenStates
        }
    }

    fileprivate class VisionModel: Module {

        @ModuleInfo(key: "patch_embed") var patchEmbed: QwenVL.PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryPositionEmbedding: QwenVL.VisionRotaryEmbedding
        @ModuleInfo(key: "blocks") var blocks: [Qwen2VLVisionBlock]
        @ModuleInfo(key: "merger") var patchMerger: PatchMerger

        let spatialMergeSize: Int

        public init(_ config: Qwen2VLConfiguration.VisionConfiguration) {
            self.spatialMergeSize = config.spatialMergeSize

            self._patchEmbed.wrappedValue = QwenVL.PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                embedDimensions: config.embedDimensions)

            let headDimensions = config.embedDimensions / config.numHeads
            self._rotaryPositionEmbedding.wrappedValue = QwenVL.VisionRotaryEmbedding(
                dimensions: headDimensions / 2, theta: 10_000)

            self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
                Qwen2VLVisionBlock(config)
            }
            self._patchMerger.wrappedValue = PatchMerger(
                dimensions: config.hiddenSize, contextDimensions: config.embedDimensions,
                spatialMergeSize: 2)
        }

        func rotaryPositionEmbedding(_ frames: [THW]) -> MLXArray {
            var positionIds = [MLXArray]()

            for row in frames {
                let (t, h, w) = row.values

                var hposIds = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hposIds = repeated(hposIds, count: w, axis: 1)
                hposIds =
                    hposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                var wposIds = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wposIds = repeated(wposIds, count: h, axis: 0)
                wposIds =
                    wposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                let stackedPosIds = stacked([hposIds, wposIds], axis: -1)
                positionIds.append(tiled(stackedPosIds, repetitions: [t, 1]))
            }

            let indices = concatenated(positionIds, axis: 0)
            let maxFrameSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 0
            let rotaryPositionEmbedFull = rotaryPositionEmbedding(sequenceLength: maxFrameSize)[
                indices]

            return rotaryPositionEmbedFull.reshaped(indices.dim(0), -1)
        }

        public func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            let rotaryPositionEmbedding = rotaryPositionEmbedding(frames)

            let batchSize = frames.count

            for block in blocks {
                hiddenStates = block(
                    hiddenStates, frames: frames,
                    rotaryPositionEmbedding: rotaryPositionEmbedding)
            }

            return patchMerger(hiddenStates)
        }

        private func isMLXWeight(_ array: MLXArray) -> Bool {
            if array.ndim != 4, array.ndim != 5 {
                return false
            }

            if array.dim(-1) == 3 {
                return true
            }

            let (outChannels, kH, kW) = (array.dim(1), array.dim(2), array.dim(3))
            return outChannels >= kH && outChannels >= kW && kH == kW
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_id") {
                    // Remove unused position_ids
                    continue
                } else if k.contains("patch_embed.proj.weight") {
                    // PyTorch conv2d weight tensors have shape:
                    //   [B, out_channels, in_channels, kH, KW]
                    // MLX conv2d expects the weight be of shape:
                    //   [B, out_channels, kH, KW, in_channels]
                    if isMLXWeight(v) {
                        sanitizedWeights[k] = v
                    } else {
                        sanitizedWeights[k] = v.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Processor

/// Qwen2VL VLM `UserInputProcessor`.
///
/// This is meant to be used with ``Qwen2VL`` and is typically created by ``VLMModelFactory``.
public class Qwen2VLProcessor: UserInputProcessor {
    public var config: Qwen2VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Qwen2VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        // first apply the user requested resizing, etc. if any
        let images = images.map { MediaProcessing.apply($0, processing: processing) }

        // image_processing_qwen2_vl._preprocess

        let size = images[0].extent.size
        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
            height: Int(size.height), width: Int(size.width),
            factor: config.patchSize * config.mergeSize,
            minPixels: config.minPixels, maxPixels: config.maxPixels)
        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)
        print("resizedWidth: \(resizedSize.width), resizedHeight: \(resizedSize.height)")
        print("patcheSize: \(config.patchSize), mergeSize: \(config.mergeSize), minPixels: \(config.minPixels), maxPixels: \(config.maxPixels)")

        let processedImages = try images.map { image in
            preprocess(image: image, resizedSize: resizedSize).asMLXArray()
        }

        return try QwenVL.patchify(
            images: processedImages, mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(messages: messages)

        // Text-only input
        if input.images.isEmpty, input.videos.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        // Process images if any
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated, frames: imagePixelsAndFrames.map { $0.1 })
            if let imageFrames = processedImage?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens, frames: imageFrames, paddingToken: "<|image_pad|>",
                    mergeSize: config.mergeSize, tokenizer: tokenizer)
            }
        }

        // Process videos if any
        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var videosAsImageSequences = [[MLXArray]]()
            var resizedSize: CGSize = .zero
            for video in input.videos {
                let imageSequence = try await MediaProcessing.asProcessedSequence(
                    video.asAVAsset(), maxFrames: config.maxFrames, targetFPS: { _ in config.fps }
                ) { frame in
                    // first apply the user requested resizing, etc. if any
                    let resizedImage = MediaProcessing.apply(
                        frame.frame, processing: input.processing)
                    if resizedSize == .zero {
                        let size = resizedImage.extent.size
                        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
                            height: Int(size.height), width: Int(size.width),
                            factor: config.patchSize * config.mergeSize,
                            minPixels: config.minPixels, maxPixels: config.maxPixels)
                        resizedSize = CGSize(width: resizedWidth, height: resizedHeight)
                    }
                    print("resizedWidth: \(resizedSize.width), resizedHeight: \(resizedSize.height)")
                    let processedImage = preprocess(image: resizedImage, resizedSize: resizedSize)
                    return VideoFrame(frame: processedImage, timeStamp: frame.timeStamp)
                }
                videosAsImageSequences.append(imageSequence.frames)
            }
            let videoPixelsAndFrames = try videosAsImageSequences.map {
                try QwenVL.patchify(
                    images: $0, mergeSize: config.mergeSize, patchSize: config.patchSize,
                    temporalPatchSize: config.temporalPatchSize)
            }
            let videoPixelsConcatenated = concatenated(videoPixelsAndFrames.map { $0.0 })
            processedVideo = LMInput.ProcessedVideo(
                pixels: videoPixelsConcatenated, frames: videoPixelsAndFrames.map { $0.1 })
            if let videoFrames = processedVideo?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens, frames: videoFrames, paddingToken: "<|video_pad|>",
                    mergeSize: config.mergeSize, tokenizer: tokenizer)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo)
    }

    /// Prepare input with frame specification for selective video processing
    /// 
    /// This method allows you to control which frames from videos are processed
    /// during the preprocessing stage, enabling more efficient video analysis.
    /// 
    /// - Parameter input: The user input containing text, images, and/or videos
    /// - Parameter frameSpecification: Which frames to process from videos
    /// - Returns: The prepared LMInput with selective frame processing
    /// - Throws: VLMError if video processing fails
    public func prepareWithFrameSpecification(input: UserInput, frameSpecification: Qwen2VL.FrameSpecification) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(messages: messages)

        // Text-only input
        if input.images.isEmpty, input.videos.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        // Process images if any
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated, frames: imagePixelsAndFrames.map { $0.1 })
            if let imageFrames = processedImage?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens, frames: imageFrames, paddingToken: "<|image_pad|>",
                    mergeSize: config.mergeSize, tokenizer: tokenizer)
            }
        }

        // Process videos with frame specification
        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var videosAsImageSequences = [[MLXArray]]()
            var resizedSize: CGSize = .zero
            
            for video in input.videos {
                let imageSequence: [MLXArray]
                
                switch frameSpecification {
                case .allFrames:
                    // Process all frames as before
                    imageSequence = try await MediaProcessing.asProcessedSequence(
                        video.asAVAsset(), maxFrames: config.maxFrames, targetFPS: { _ in config.fps }
                    ) { frame in
                        let resizedImage = MediaProcessing.apply(frame.frame, processing: input.processing)
                        if resizedSize == .zero {
                            let size = resizedImage.extent.size
                            let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
                                height: Int(size.height), width: Int(size.width),
                                factor: config.patchSize * config.mergeSize,
                                minPixels: config.minPixels, maxPixels: config.maxPixels)
                            resizedSize = CGSize(width: resizedWidth, height: resizedHeight)
                        }
                        let processedImage = preprocess(image: resizedImage, resizedSize: resizedSize)
                        return VideoFrame(frame: processedImage, timeStamp: frame.timeStamp)
                    }.frames
                    
                case .frameNumbers(let frameNumbers):
                    // Process only specified frame numbers
                    let asset = video.asAVAsset()
                    let duration = try await asset.load(.duration)
                    let durationSeconds = CMTimeGetSeconds(duration)
                    let maxFrame = Int(durationSeconds * config.fps) - 1
                    
                    let validFrameNumbers = frameNumbers.filter { $0 >= 0 && $0 <= maxFrame }
                    if validFrameNumbers.isEmpty {
                        throw NSError(domain: "VideoProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "No valid frame numbers provided"])
                    }
                    
                    imageSequence = try await processSpecificFrames(
                        asset: asset,
                        frameNumbers: validFrameNumbers,
                        processing: input.processing,
                        resizedSize: &resizedSize
                    )
                    
                case .timestamps(let timestamps):
                    // Process frames at specific timestamps
                    let asset = video.asAVAsset()
                    let fps = config.fps
                    let frameNumbers = timestamps.map { Int($0 * fps) }
                    
                    imageSequence = try await processSpecificFrames(
                        asset: asset,
                        frameNumbers: frameNumbers,
                        processing: input.processing,
                        resizedSize: &resizedSize
                    )
                }
                
                videosAsImageSequences.append(imageSequence)
            }
            
            let videoPixelsAndFrames = try videosAsImageSequences.map {
                try QwenVL.patchify(
                    images: $0, mergeSize: config.mergeSize, patchSize: config.patchSize,
                    temporalPatchSize: config.temporalPatchSize)
            }
            let videoPixelsConcatenated = concatenated(videoPixelsAndFrames.map { $0.0 })
            processedVideo = LMInput.ProcessedVideo(
                pixels: videoPixelsConcatenated, frames: videoPixelsAndFrames.map { $0.1 })
            if let videoFrames = processedVideo?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens, frames: videoFrames, paddingToken: "<|video_pad|>",
                    mergeSize: config.mergeSize, tokenizer: tokenizer)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo)
    }

    /// Helper method to process specific frames from a video asset
    private func processSpecificFrames(
        asset: AVAsset,
        frameNumbers: [Int],
        processing: UserInput.Processing,
        resizedSize: inout CGSize
    ) async throws -> [MLXArray] {
        var imageSequence: [MLXArray] = []
        
        for frameNumber in frameNumbers {
            let timestamp = TimeInterval(frameNumber) / config.fps
            let frameImage = try await extractFrameFromAsset(asset, at: timestamp)
            
            let resizedImage = MediaProcessing.apply(frameImage, processing: processing)
            if resizedSize == .zero {
                let size = resizedImage.extent.size
                let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
                    height: Int(size.height), width: Int(size.width),
                    factor: config.patchSize * config.mergeSize,
                    minPixels: config.minPixels, maxPixels: config.maxPixels)
                resizedSize = CGSize(width: resizedWidth, height: resizedHeight)
            }
            let processedImage = preprocess(image: resizedImage, resizedSize: resizedSize)
            imageSequence.append(processedImage.asMLXArray())
        }
        
        return imageSequence
    }

    /// Helper function to extract a single frame from an asset at a specific timestamp
    private func extractFrameFromAsset(_ asset: AVAsset, at timestamp: TimeInterval) async throws -> CIImage {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        
        let cmTime = CMTime(seconds: timestamp, preferredTimescale: 600)
        
        do {
            let cgImage = try await generator.image(at: cmTime).image
            return CIImage(cgImage: cgImage, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
        } catch {
            throw NSError(
                domain: "VideoProcessing", 
                code: -1, 
                userInfo: [NSLocalizedDescriptionKey: "Failed to extract frame at timestamp \(timestamp): \(error.localizedDescription)"]
            )
        }
    }
}

// MARK: - Model

/// Qwen2VL VLM
///
/// This is typically created by ``VLMModelFactory``.
public class Qwen2VL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel

    public let config: Qwen2VLConfiguration

    public var vocabularySize: Int { config.baseConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: Qwen2VLConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, frames: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let frames else {
            return languageModel.model.embedTokens(inputIds[.newAxis, .ellipsis])
        }

        // Get the input embeddings from the language model
        let inputEmbeds = languageModel.model.embedTokens(inputIds)

        // Get the ouptut hidden states from the vision model
        var hiddenStates = self.visionModel(pixelValues, frames: frames)

        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        // Insert special image tokens in the input_ids
        return QwenVL.mergeInputIdsWithImageFeatures(
            inputIds: inputIds, inputEmbeds: inputEmbeds, imageFeatures: hiddenStates,
            imageTokenId: config.baseConfiguration.imageTokenId,
            videoTokenId: config.baseConfiguration.videoTokenId)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let dtype = visionModel.patchEmbed.proj.weight.dtype

        // Process both images and videos together
        var allPixels: MLXArray?
        var allFrames: [THW] = []

        if let imagePixels = input.image?.pixels, let imageFrames = input.image?.frames {
            allPixels = imagePixels.asType(dtype)
            allFrames.append(contentsOf: imageFrames)
        }

        if let videoPixels = input.video?.pixels, let videoFrames = input.video?.frames {
            if allPixels == nil {
                allPixels = videoPixels.asType(dtype)
            } else {
                allPixels = concatenated([allPixels!, videoPixels.asType(dtype)])
            }
            allFrames.append(contentsOf: videoFrames)
        }

        let inputEmbeddings = self.inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: allPixels,
            frames: allFrames.isEmpty ? nil : allFrames)

        let result = languageModel(nil, cache: cache, inputEmbedding: inputEmbeddings)

        return .logits(result)
    }

    /// Prepare the model with frame specification for selective video processing
    /// 
    /// This method allows you to control which frames from videos are processed
    /// during model inference, enabling more efficient and targeted video analysis.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen2VL(config)
    /// let input = LMInput(...) // with video content
    /// 
    /// // Process only specific frames
    /// let result = try model.prepareWithFrameSpecification(
    ///     input, 
    ///     cache: cache, 
    ///     windowSize: nil,
    ///     frameSpecification: .frameNumbers([0, 10, 20, 30])
    /// )
    /// ```
    /// 
    /// - Parameter input: The input containing text, images, and/or videos
    /// - Parameter cache: The KV cache for attention
    /// - Parameter windowSize: Optional window size for sliding window attention
    /// - Parameter frameSpecification: Which frames to process from videos
    /// - Returns: The prepare result containing logits or tokens
    /// - Throws: VLMError if video processing fails
    public func prepareWithFrameSpecification(_ input: LMInput, cache: [any KVCache], windowSize: Int?, frameSpecification: FrameSpecification) throws -> PrepareResult {
        let dtype = visionModel.patchEmbed.proj.weight.dtype

        // Process both images and videos together
        var allPixels: MLXArray?
        var allFrames: [THW] = []

        if let imagePixels = input.image?.pixels, let imageFrames = input.image?.frames {
            allPixels = imagePixels.asType(dtype)
            allFrames.append(contentsOf: imageFrames)
        }

        // Process videos with frame specification
        if let videoPixels = input.video?.pixels, let videoFrames = input.video?.frames {
            // Apply frame specification to video frames
            let (filteredPixels, filteredFrames) = try applyFrameSpecificationToVideo(
                pixels: videoPixels,
                frames: videoFrames,
                frameSpecification: frameSpecification
            )
            
            if allPixels == nil {
                allPixels = filteredPixels.asType(dtype)
            } else {
                allPixels = concatenated([allPixels!, filteredPixels.asType(dtype)])
            }
            allFrames.append(contentsOf: filteredFrames)
        }

        let inputEmbeddings = self.inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: allPixels,
            frames: allFrames.isEmpty ? nil : allFrames)

        let result = languageModel(nil, cache: cache, inputEmbedding: inputEmbeddings)

        return .logits(result)
    }

    /// Apply frame specification to video pixels and frames
    private func applyFrameSpecificationToVideo(
        pixels: MLXArray,
        frames: [THW],
        frameSpecification: FrameSpecification
    ) throws -> (MLXArray, [THW]) {
        switch frameSpecification {
        case .allFrames:
            return (pixels, frames)
            
        case .frameNumbers(let frameNumbers):
            // Validate frame numbers
            let maxFrame = frames.count - 1
            let validFrameNumbers = frameNumbers.filter { $0 >= 0 && $0 <= maxFrame }
            
            if validFrameNumbers.isEmpty {
                throw NSError(domain: "VideoProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "No valid frame numbers provided"])
            }
            
            // Extract specified frames
            var selectedPixels: [MLXArray] = []
            var selectedFrames: [THW] = []
            
            for frameNumber in validFrameNumbers {
                // Calculate the start and end indices for this frame in the pixels array
                let frameStartIndex = frames.prefix(frameNumber).reduce(0) { $0 + $1.t }
                let frameEndIndex = frameStartIndex + frames[frameNumber].t
                
                let framePixels = pixels[frameStartIndex..<frameEndIndex]
                selectedPixels.append(framePixels)
                selectedFrames.append(frames[frameNumber])
            }
            
            return (concatenated(selectedPixels), selectedFrames)
            
        case .timestamps(let timestamps):
            // Convert timestamps to frame numbers based on frame duration
            // This is a simplified approach - in practice, you'd need to know the video FPS
            let fps = 2.0 // Default FPS for video processing
            let frameNumbers = timestamps.map { Int($0 * fps) }
            
            return try applyFrameSpecificationToVideo(
                pixels: pixels,
                frames: frames,
                frameSpecification: .frameNumbers(frameNumbers)
            )
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        visionModel.sanitize(
            weights:
                Dictionary(
                    uniqueKeysWithValues: weights.map { key, value in
                        var key = key
                        if !key.contains("vision_tower") {
                            key = key.replacingOccurrences(of: "visual", with: "vision_tower")
                        }
                        if !key.contains("language_model") {
                            key = key.replacingOccurrences(
                                of: "model", with: "language_model.model")
                            key = key.replacingOccurrences(
                                of: "lm_head", with: "language_model.lm_head")
                        }

                        return (key, value)
                    })
        )
    }

    /// Extract patch embeddings from a single image
    /// 
    /// This function preprocesses the image and applies patch embedding to get the initial hidden states.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen2VL(config)
    /// let processorConfig = Qwen2VLProcessorConfiguration(...)
    /// let image = CIImage(contentsOf: imageURL)!
    /// let patchEmbeddings = try model.extractPatchEmbeddings(from: image, processorConfig: processorConfig)
    /// // patchEmbeddings shape: [numPatches, embedDimensions]
    /// ```
    /// 
    /// - Parameter image: The input image as a CIImage
    /// - Parameter processing: Optional processing parameters for resizing, etc.
    /// - Parameter processorConfig: The processor configuration containing minPixels and maxPixels
    /// - Returns: The patch embeddings as an MLXArray
    /// - Throws: VLMError if image processing fails
    public func extractPatchEmbeddings(
        from image: CIImage, 
        processing: UserInput.Processing? = nil,
        processorConfig: Qwen2VLProcessorConfiguration
    ) throws -> MLXArray {
        // Apply user processing if provided
        let processedImage = MediaProcessing.apply(image, processing: processing)
        
        // Calculate target size for resizing
        let size = processedImage.extent.size
        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
            height: Int(size.height), width: Int(size.width),
            factor: processorConfig.patchSize * processorConfig.mergeSize,
            minPixels: processorConfig.minPixels, maxPixels: processorConfig.maxPixels)

        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)
        print("Resized size: \(resizedSize)")
        print("patcheSize: \(processorConfig.patchSize), mergeSize: \(processorConfig.mergeSize), minPixels: \(processorConfig.minPixels), maxPixels: \(processorConfig.maxPixels)")

        // Preprocess the image (resize, normalize)
        let normalizedImage = processedImage
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: (0.48145466, 0.4578275, 0.40821073), std: (0.26862954, 0.26130258, 0.27577711))
        
        // Convert to MLXArray
        let imageArray = normalizedImage.asMLXArray()
        
        // Patchify the image
        let (pixelValues, frames) = try QwenVL.patchify(
            images: [imageArray], 
            mergeSize: processorConfig.mergeSize, 
            patchSize: processorConfig.patchSize,
            temporalPatchSize: processorConfig.temporalPatchSize
        )
        
        // Convert to the correct data type for the patch embed
        let dtype = visionModel.patchEmbed.proj.weight.dtype
        let typedPixelValues = pixelValues.asType(dtype)
        
        // Apply patch embedding: var hiddenStates = patchEmbed(hiddenStates)
        let hiddenStates = visionModel.patchEmbed(typedPixelValues)
        
        // Print the dimensions of the hidden states
        print("Hidden states dimensions: \(hiddenStates.shape)")
        print("Hidden states size: \(hiddenStates.size)")
        print("Hidden states data type: \(hiddenStates.dtype)")
        
        // Calculate size in bytes based on the actual data type
        let bytesPerElement: Int
        switch hiddenStates.dtype {
        case .float16:
            bytesPerElement = 2
        case .float32:
            bytesPerElement = 4
        case .float64:
            bytesPerElement = 8
        default:
            bytesPerElement = 4 // fallback
        }
        print("Hidden states size in bytes: \(hiddenStates.size * bytesPerElement)")
        
        return hiddenStates
    }

    /// Extract and mean-pool patch embeddings from a single image into a 1D vector
    /// 
    /// This function preprocesses the image, applies patch embedding, and then mean-pools
    /// the resulting patch embeddings into a single 1D feature vector.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen2VL(config)
    /// let processorConfig = Qwen2VLProcessorConfiguration(...)
    /// let image = CIImage(contentsOf: imageURL)!
    /// let pooledEmbeddings = try model.extractAndPoolEmbeddings(from: image, processorConfig: processorConfig)
    /// // pooledEmbeddings shape: [embedDimensions] (1D vector)
    /// ```
    /// 
    /// - Parameter image: The input image as a CIImage
    /// - Parameter processing: Optional processing parameters for resizing, etc.
    /// - Parameter processorConfig: The processor configuration containing minPixels and maxPixels
    /// - Returns: The mean-pooled embeddings as a 1D MLXArray
    /// - Throws: VLMError if image processing fails
    public func extractAndPoolEmbeddings(
        from image: CIImage, 
        processing: UserInput.Processing? = nil,
        processorConfig: Qwen2VLProcessorConfiguration
    ) throws -> MLXArray {
        // Get the patch embeddings
        let patchEmbeddings = try extractPatchEmbeddings(
            from: image, 
            processing: processing, 
            processorConfig: processorConfig
        )
        
        // Mean-pool across the patch dimension (first dimension)
        let pooledEmbeddings = mean(patchEmbeddings, axis: 0)
        
        // Print information about the pooled embeddings
        print("Pooled embeddings dimensions: \(pooledEmbeddings.shape)")
        print("Pooled embeddings size: \(pooledEmbeddings.size)")
        print("Pooled embeddings data type: \(pooledEmbeddings.dtype)")
        
        // Calculate size in bytes based on the actual data type
        let bytesPerElement: Int
        switch pooledEmbeddings.dtype {
        case .float16:
            bytesPerElement = 2
        case .float32:
            bytesPerElement = 4
        case .float64:
            bytesPerElement = 8
        default:
            bytesPerElement = 4 // fallback
        }
        print("Pooled embeddings size in bytes: \(pooledEmbeddings.size * bytesPerElement)")
        
        return pooledEmbeddings
    }

    /// Calculate cosine distance between two mean-pooled embeddings
    /// 
    /// This function computes the cosine distance between two 1D feature vectors
    /// obtained from mean-pooled patch embeddings. Cosine distance is 1 - cosine_similarity.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen2VL(config)
    /// let processorConfig = Qwen2VLProcessorConfiguration(...)
    /// let image1 = CIImage(contentsOf: imageURL1)!
    /// let image2 = CIImage(contentsOf: imageURL2)!
    /// 
    /// let embedding1 = try model.extractAndPoolEmbeddings(from: image1, processorConfig: processorConfig)
    /// let embedding2 = try model.extractAndPoolEmbeddings(from: image2, processorConfig: processorConfig)
    /// let distance = model.cosineDistance(embedding1, embedding2)
    /// // distance is a value between 0 and 2, where 0 means identical
    /// ```
    /// 
    /// - Parameter embedding1: First 1D embedding vector
    /// - Parameter embedding2: Second 1D embedding vector
    /// - Returns: Cosine distance value between 0 and 2
    public func cosineDistance(_ embedding1: MLXArray, _ embedding2: MLXArray) -> Float {
        // Ensure both embeddings are 1D
        let vec1 = embedding1.flattened()
        let vec2 = embedding2.flattened()
        
        // Calculate dot product
        let dotProduct = sum(vec1 * vec2)
        
        // Calculate magnitudes
        let magnitude1 = sqrt(sum(vec1 * vec1))
        let magnitude2 = sqrt(sum(vec2 * vec2))
        
        // Calculate cosine similarity first
        let similarity = dotProduct / (magnitude1 * magnitude2)
        
        // Convert to cosine distance (1 - similarity)
        let distance = 1.0 - similarity.item() as Float
        
        // Handle potential NaN
        return distance.isNaN ? 2.0 : distance
    }

    /// Extract patch embeddings from a video by processing each frame
    /// 
    /// This function processes each frame of a video, extracts patch embeddings,
    /// and returns an array of patch embeddings for each frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen2VL(config)
    /// let processorConfig = Qwen2VLProcessorConfiguration(...)
    /// let videoURL = URL(fileURLWithPath: "video.mp4")
    /// let frameEmbeddings = try model.extractVideoPatchEmbeddings(from: videoURL, processorConfig: processorConfig)
    /// // frameEmbeddings is an array of MLXArray, one for each frame
    /// ```
    /// 
    /// - Parameter videoURL: The URL of the video file
    /// - Parameter processing: Optional processing parameters for resizing, etc.
    /// - Parameter processorConfig: The processor configuration containing minPixels and maxPixels
    /// - Returns: Array of patch embeddings for each frame
    /// - Throws: VLMError if video processing fails
    public func extractVideoPatchEmbeddings(
        from videoURL: URL,
        processing: UserInput.Processing? = nil,
        processorConfig: Qwen2VLProcessorConfiguration
    ) async throws -> [MLXArray] {
        // Extract CIImage frames from video
        let ciImages = try await MediaProcessing.asCIImageSequence(
            AVAsset(url: videoURL), 
            samplesPerSecond: Int(processorConfig.fps)
        )
        
        var frameEmbeddings: [MLXArray] = []
        
        // Process each frame
        for (index, frameImage) in ciImages.enumerated() {
            print("Processing frame \(index + 1)/\(ciImages.count)")
            
            let frameEmbedding = try extractPatchEmbeddings(
                from: frameImage,
                processing: processing,
                processorConfig: processorConfig
            )
            
            frameEmbeddings.append(frameEmbedding)
        }
        
        print("Successfully extracted patch embeddings for \(frameEmbeddings.count) frames")
        return frameEmbeddings
    }

    /// Extract and mean-pool embeddings from a video by processing each frame
    /// 
    /// This function processes each frame of a video, extracts patch embeddings,
    /// mean-pools them, and returns an array of 1D feature vectors for each frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen2VL(config)
    /// let processorConfig = Qwen2VLProcessorConfiguration(...)
    /// let videoURL = URL(fileURLWithPath: "video.mp4")
    /// let frameEmbeddings = try model.extractAndPoolVideoEmbeddings(from: videoURL, processorConfig: processorConfig)
    /// // frameEmbeddings is an array of 1D MLXArray, one for each frame
    /// ```
    /// 
    /// - Parameter videoURL: The URL of the video file
    /// - Parameter processing: Optional processing parameters for resizing, etc.
    /// - Parameter processorConfig: The processor configuration containing minPixels and maxPixels
    /// - Returns: Array of mean-pooled embeddings for each frame
    /// - Throws: VLMError if video processing fails
    public func extractAndPoolVideoEmbeddings(
        from videoURL: URL,
        processing: UserInput.Processing? = nil,
        processorConfig: Qwen2VLProcessorConfiguration
    ) async throws -> [MLXArray] {
        // Extract CIImage frames from video
        let ciImages = try await MediaProcessing.asCIImageSequence(
            AVAsset(url: videoURL), 
            samplesPerSecond: Int(processorConfig.fps)
        )
        
        var frameEmbeddings: [MLXArray] = []
        
        // Process each frame
        for (index, frameImage) in ciImages.enumerated() {
            print("Processing frame \(index + 1)/\(ciImages.count)")
            
            let frameEmbedding = try extractAndPoolEmbeddings(
                from: frameImage,
                processing: processing,
                processorConfig: processorConfig
            )
            
            frameEmbeddings.append(frameEmbedding)
        }
        
        print("Successfully extracted and mean-pooled embeddings for \(frameEmbeddings.count) frames")
        return frameEmbeddings
    }

    /// Calculate cosine distance between each frame and the first frame as reference
    /// 
    /// This function extracts mean-pooled embeddings from each frame of a video
    /// and calculates cosine distance between each frame and the first frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen2VL(config)
    /// let processorConfig = Qwen2VLProcessorConfiguration(...)
    /// let videoURL = URL(fileURLWithPath: "video.mp4")
    /// let distances = try model.calculateVideoFrameDistances(from: videoURL, processorConfig: processorConfig)
    /// // distances is an array of Float values, one for each frame
    /// ```
    /// 
    /// - Parameter videoURL: The URL of the video file
    /// - Parameter processing: Optional processing parameters for resizing, etc.
    /// - Parameter processorConfig: The processor configuration containing minPixels and maxPixels
    /// - Returns: Array of cosine distance values for each frame (first frame will be 0.0)
    /// - Throws: VLMError if video processing fails
    public func calculateVideoFrameDistances(
        from videoURL: URL,
        processing: UserInput.Processing? = nil,
        processorConfig: Qwen2VLProcessorConfiguration
    ) async throws -> [Float] {
        // Extract mean-pooled embeddings from each frame
        let frameEmbeddings = try await extractAndPoolVideoEmbeddings(
            from: videoURL,
            processing: processing,
            processorConfig: processorConfig
        )
        
        guard !frameEmbeddings.isEmpty else {
            throw NSError(domain: "VideoProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "No frames extracted from video"])
        }
        
        let referenceEmbedding = frameEmbeddings[0]
        var similarities: [Float] = []
        
        // Calculate distance between each frame and the reference frame
        for (index, frameEmbedding) in frameEmbeddings.enumerated() {
            let distance = cosineDistance(referenceEmbedding, frameEmbedding)
            similarities.append(distance)
            
            print("Frame \(index + 1) distance to reference: \(distance)")
        }
        
        print("Successfully calculated similarities for \(similarities.count) frames")
        return similarities
    }

    /// Detect scene changes in a video using cosine distance threshold
    /// 
    /// This function analyzes each frame of a video and detects scene changes
    /// by comparing each frame to a reference frame. When distance exceeds
    /// the threshold, it marks a scene change and updates the reference frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen2VL(config)
    /// let processorConfig = Qwen2VLProcessorConfiguration(...)
    /// let videoURL = URL(fileURLWithPath: "video.mp4")
    /// let sceneChanges = try model.detectSceneChanges(from: videoURL, threshold: 0.1, processorConfig: processorConfig)
    /// // sceneChanges contains frame indices where scene changes occur
    /// ```
    /// 
    /// - Parameter videoURL: The URL of the video file
    /// - Parameter threshold: Cosine distance threshold for scene change detection (default: 0.1)
    /// - Parameter processing: Optional processing parameters for resizing, etc.
    /// - Parameter processorConfig: The processor configuration containing minPixels and maxPixels
    /// - Returns: Array of frame indices where scene changes occur (including frame 0)
    /// - Throws: VLMError if video processing fails
    public func detectSceneChanges(
        from videoURL: URL,
        threshold: Float = 0.1,
        minSceneDuration: TimeInterval = 2.0, // Minimum scene duration in seconds
        maxSceneDuration: TimeInterval = 15.0, // Maximum scene duration in seconds
        processing: UserInput.Processing? = nil,
        processorConfig: Qwen2VLProcessorConfiguration
    ) async throws -> [(frameIndex: Int, timestamp: TimeInterval)] {
        let startTime = Date()
        
        // Extract CIImage frames from video at 2 FPS for scene detection
        let ciImages = try await MediaProcessing.asCIImageSequence(
            AVAsset(url: videoURL), 
            samplesPerSecond: 2
        )
        
        var frameEmbeddings: [MLXArray] = []
        var frameTimestamps: [TimeInterval] = []
        
        // Process each frame
        for (index, frameImage) in ciImages.enumerated() {
            print("Processing frame \(index + 1)/\(ciImages.count)")
            
            let frameEmbedding = try extractAndPoolEmbeddings(
                from: frameImage,
                processing: processing,
                processorConfig: processorConfig
            )
            
            frameEmbeddings.append(frameEmbedding)
            
            // Calculate timestamp for this frame (2 FPS = 0.5 seconds per frame)
            let timestamp = TimeInterval(index) * 0.5
            frameTimestamps.append(timestamp)
        }
        
        guard !frameEmbeddings.isEmpty else {
            throw NSError(domain: "VideoProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "No frames extracted from video"])
        }
        
        var sceneChanges: [(frameIndex: Int, timestamp: TimeInterval)] = [(0, 0.0)] // Always include frame 0 as first scene
        var currentReferenceEmbedding = frameEmbeddings[0]
        
        print("Scene change detection with threshold: \(threshold), min scene duration: \(minSceneDuration)s, max scene duration: \(maxSceneDuration)s")
        print("Frame 0 (0.0s): Starting new scene (reference frame)")
        
        // Analyze each frame starting from frame 1
        for frameIndex in 1..<frameEmbeddings.count {
            let currentEmbedding = frameEmbeddings[frameIndex]
            let distance = cosineDistance(currentReferenceEmbedding, currentEmbedding)
            let timestamp = frameTimestamps[frameIndex]
            let timeSinceLastScene = timestamp - sceneChanges.last!.timestamp
            
            print("Frame \(frameIndex) (\(String(format: "%.1f", timestamp))s): distance to reference = \(String(format: "%.4f", distance)), time since last scene: \(String(format: "%.1f", timeSinceLastScene))s")
            
            var sceneChangeDetected = false
            var sceneChangeReason = ""
            
            // Check if maximum scene duration has been exceeded
            if timeSinceLastScene >= maxSceneDuration {
                sceneChangeDetected = true
                sceneChangeReason = "max duration exceeded"
                print("Frame \(frameIndex) (\(String(format: "%.1f", timestamp))s): FORCED SCENE CHANGE - Max duration exceeded (\(String(format: "%.1f", timeSinceLastScene))s >= \(maxSceneDuration)s)")
            }
            // Check if distance threshold is exceeded and minimum duration is met
            else if distance > threshold && timeSinceLastScene >= minSceneDuration {
                sceneChangeDetected = true
                sceneChangeReason = "distance threshold"
                print("Frame \(frameIndex) (\(String(format: "%.1f", timestamp))s): SCENE CHANGE DETECTED - Distance threshold exceeded (duration: \(String(format: "%.1f", timeSinceLastScene))s)")
            }
            // Check if distance threshold is exceeded but minimum duration is not met
            else if distance > threshold && timeSinceLastScene < minSceneDuration {
                print("Frame \(frameIndex) (\(String(format: "%.1f", timestamp))s): Scene change ignored - too short (duration: \(String(format: "%.1f", timeSinceLastScene))s < \(minSceneDuration)s)")
            }
            
            if sceneChangeDetected {
                sceneChanges.append((frameIndex: frameIndex, timestamp: timestamp))
                currentReferenceEmbedding = currentEmbedding
                print("Frame \(frameIndex) (\(String(format: "%.1f", timestamp))s): SCENE CHANGE APPLIED - \(sceneChangeReason) (duration: \(String(format: "%.1f", timeSinceLastScene))s)")
            }
        }
        
        let endTime = Date()
        let duration = endTime.timeIntervalSince(startTime)
        
        print("Scene change detection complete!")
        print("Total scenes detected: \(sceneChanges.count)")
        print("Scene changes at frames and timestamps:")
        for (frameIndex, timestamp) in sceneChanges {
            print("  Frame \(frameIndex): \(String(format: "%.1f", timestamp))s")
        }
        print("Total processing time: \(String(format: "%.2f", duration)) seconds")
        print("Average time per frame: \(String(format: "%.3f", duration / Double(frameEmbeddings.count))) seconds")
        
        return sceneChanges
    }

    /// Helper function to extract a single frame from an asset at a specific timestamp
    private func extractFrameFromAsset(_ asset: AVAsset, at timestamp: TimeInterval) async throws -> CIImage {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        
        let cmTime = CMTime(seconds: timestamp, preferredTimescale: 600)
        
        do {
            let cgImage = try await generator.image(at: cmTime).image
            return CIImage(cgImage: cgImage, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
        } catch {
            throw NSError(
                domain: "VideoProcessing", 
                code: -1, 
                userInfo: [NSLocalizedDescriptionKey: "Failed to extract frame at timestamp \(timestamp): \(error.localizedDescription)"]
            )
        }
    }

    /// Frame specification for selective video processing
    public enum FrameSpecification {
        /// Process specific frame numbers (0-based indexing)
        case frameNumbers([Int])
        /// Process frames at specific timestamps (in seconds)
        case timestamps([TimeInterval])
        /// Process all frames (default behavior)
        case allFrames
    }

    /// Extract and mean-pool embeddings from specific frames of a video
    /// 
    /// This function processes only the specified frames of a video, extracts patch embeddings,
    /// mean-pools them, and returns an array of 1D feature vectors for each specified frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen2VL(config)
    /// let processorConfig = Qwen2VLProcessorConfiguration(...)
    /// let videoURL = URL(fileURLWithPath: "video.mp4")
    /// 
    /// // Process specific frame numbers
    /// let frameEmbeddings = try model.extractAndPoolVideoEmbeddings(
    ///     from: videoURL, 
    ///     frameSpecification: .frameNumbers([0, 10, 20, 30]),
    ///     processorConfig: processorConfig
    /// )
    /// 
    /// // Process frames at specific timestamps
    /// let frameEmbeddings = try model.extractAndPoolVideoEmbeddings(
    ///     from: videoURL, 
    ///     frameSpecification: .timestamps([0.0, 5.0, 10.0, 15.0]),
    ///     processorConfig: processorConfig
    /// )
    /// ```
    /// 
    /// - Parameter videoURL: The URL of the video file
    /// - Parameter frameSpecification: Which frames to process (frame numbers, timestamps, or all frames)
    /// - Parameter processing: Optional processing parameters for resizing, etc.
    /// - Parameter processorConfig: The processor configuration containing minPixels and maxPixels
    /// - Returns: Array of mean-pooled embeddings for each specified frame
    /// - Throws: VLMError if video processing fails
    public func extractAndPoolVideoEmbeddings(
        from videoURL: URL,
        frameSpecification: FrameSpecification = .allFrames,
        processing: UserInput.Processing? = nil,
        processorConfig: Qwen2VLProcessorConfiguration
    ) async throws -> [MLXArray] {
        // Get video asset and duration
        let asset = AVAsset(url: videoURL)
        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)
        
        print("Video duration: \(String(format: "%.2f", durationSeconds)) seconds")
        
        // Determine which frames to process
        let framesToProcess: [Int]
        let frameTimestamps: [TimeInterval]
        
        switch frameSpecification {
        case .frameNumbers(let frameNumbers):
            framesToProcess = frameNumbers.sorted()
            frameTimestamps = frameNumbers.map { TimeInterval($0) / processorConfig.fps }
            print("Processing specific frame numbers: \(frameNumbers)")
            
        case .timestamps(let timestamps):
            let sortedTimestamps = timestamps.sorted()
            framesToProcess = sortedTimestamps.map { Int($0 * processorConfig.fps) }
            frameTimestamps = sortedTimestamps
            print("Processing frames at timestamps: \(timestamps.map { String(format: "%.2f", $0) })")
            
        case .allFrames:
            // Extract all frames as before
            let ciImages = try await MediaProcessing.asCIImageSequence(
                AVAsset(url: videoURL), 
                samplesPerSecond: Int(processorConfig.fps)
            )
            
            var frameEmbeddings: [MLXArray] = []
            
            // Process each frame
            for (index, frameImage) in ciImages.enumerated() {
                print("Processing frame \(index + 1)/\(ciImages.count)")
                
                let frameEmbedding = try extractAndPoolEmbeddings(
                    from: frameImage,
                    processing: processing,
                    processorConfig: processorConfig
                )
                
                frameEmbeddings.append(frameEmbedding)
            }
            
            print("Successfully extracted and mean-pooled embeddings for \(frameEmbeddings.count) frames")
            return frameEmbeddings
        }
        
        // Validate frame numbers
        let maxFrameNumber = Int(durationSeconds * processorConfig.fps)
        let validFrames = framesToProcess.filter { $0 >= 0 && $0 < maxFrameNumber }
        
        if validFrames.count != framesToProcess.count {
            let invalidFrames = framesToProcess.filter { $0 < 0 || $0 >= maxFrameNumber }
            print("Warning: Invalid frame numbers ignored: \(invalidFrames)")
        }
        
        guard !validFrames.isEmpty else {
            throw NSError(domain: "VideoProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "No valid frames to process"])
        }
        
        print("Processing \(validFrames.count) valid frames out of \(framesToProcess.count) requested")
        
        // Extract specific frames using MediaProcessing
        var frameEmbeddings: [MLXArray] = []
        
        for (index, frameNumber) in validFrames.enumerated() {
            let timestamp = frameTimestamps[index]
            print("Processing frame \(frameNumber) at timestamp \(String(format: "%.2f", timestamp))s (\(index + 1)/\(validFrames.count))")
            
            // Extract single frame at specific timestamp
            let frameImage = try await extractFrameFromAsset(asset, at: timestamp)
            
            let frameEmbedding = try extractAndPoolEmbeddings(
                from: frameImage,
                processing: processing,
                processorConfig: processorConfig
            )
            
            frameEmbeddings.append(frameEmbedding)
        }
        
        print("Successfully extracted and mean-pooled embeddings for \(frameEmbeddings.count) specified frames")
        return frameEmbeddings
    }

    /// Calculate cosine distance between specified frames and the first frame as reference
    /// 
    /// This function extracts mean-pooled embeddings from specified frames of a video
    /// and calculates cosine distance between each frame and the first frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen2VL(config)
    /// let processorConfig = Qwen2VLProcessorConfiguration(...)
    /// let videoURL = URL(fileURLWithPath: "video.mp4")
    /// 
    /// // Calculate distances for specific frames
    /// let distances = try model.calculateVideoFrameDistances(
    ///     from: videoURL, 
    ///     frameSpecification: .frameNumbers([0, 10, 20, 30]),
    ///     processorConfig: processorConfig
    /// )
    /// ```
    /// 
    /// - Parameter videoURL: The URL of the video file
    /// - Parameter frameSpecification: Which frames to process (frame numbers, timestamps, or all frames)
    /// - Parameter processing: Optional processing parameters for resizing, etc.
    /// - Parameter processorConfig: The processor configuration containing minPixels and maxPixels
    /// - Returns: Array of cosine distance values for each frame (first frame will be 0.0)
    /// - Throws: VLMError if video processing fails
    public func calculateVideoFrameDistances(
        from videoURL: URL,
        frameSpecification: FrameSpecification = .allFrames,
        processing: UserInput.Processing? = nil,
        processorConfig: Qwen2VLProcessorConfiguration
    ) async throws -> [Float] {
        // Extract mean-pooled embeddings from specified frames
        let frameEmbeddings = try await extractAndPoolVideoEmbeddings(
            from: videoURL,
            frameSpecification: frameSpecification,
            processing: processing,
            processorConfig: processorConfig
        )
        
        guard !frameEmbeddings.isEmpty else {
            throw NSError(domain: "VideoProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "No frames extracted from video"])
        }
        
        let referenceEmbedding = frameEmbeddings[0]
        var similarities: [Float] = []
        
        // Calculate distance between each frame and the reference frame
        for (index, frameEmbedding) in frameEmbeddings.enumerated() {
            let distance = cosineDistance(referenceEmbedding, frameEmbedding)
            similarities.append(distance)
            
            print("Frame \(index + 1) distance to reference: \(distance)")
        }
        
        print("Successfully calculated similarities for \(similarities.count) frames")
        return similarities
    }



}

// MARK: - Configuration

/// Configuration for ``Qwen2VL``
public struct Qwen2VLConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }
        public let vocabularySize: Int
        public let kvHeads: Int
        private let _maxPositionEmbeddings: Int?
        public var maxpPositionEmbeddings: Int { _maxPositionEmbeddings ?? 32768 }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 1_000_000 }
        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool { _ropeTraditional ?? false }
        public let ropeScaling: [String: StringOrNumber]?
        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? true }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case _rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case kvHeads = "num_key_value_heads"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case _tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let depth: Int
        public let embedDimensions: Int
        public let hiddenSize: Int
        public let numHeads: Int
        public let patchSize: Int
        public let mlpRatio: Float
        public let _inChannels: Int?
        public var inChannels: Int { _inChannels ?? 3 }
        public let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }
        public let spatialPatchSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int

        enum CodingKeys: String, CodingKey {
            case depth
            case embedDimensions = "embed_dim"
            case hiddenSize = "hidden_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case mlpRatio = "mlp_ratio"
            case _inChannels = "in_channels"
            case _layerNormEps = "layer_norm_eps"
            case spatialPatchSize = "spatial_patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
        }
    }

    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let imageTokenId: Int
        public let videoTokenId: Int
        public let hiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case imageTokenId = "image_token_id"
            case videoTokenId = "video_token_id"
            case hiddenSize = "hidden_size"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let baseConfiguration: BaseConfiguration

    enum CodingKeys: String, CodingKey {
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // this is a sub-dictionary
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)

        // these are overlaid in the top level
        self.textConfiguration = try TextConfiguration(from: decoder)
        self.baseConfiguration = try BaseConfiguration(from: decoder)
    }
}

/// Configuration for ``Qwen2VLProcessor``
public struct Qwen2VLProcessorConfiguration: Codable, Sendable {

    public struct Size: Codable, Sendable {
        public let maxPixels: Int
        public let minPixels: Int

        enum CodingKeys: String, CodingKey {
            case maxPixels = "max_pixels"
            case minPixels = "min_pixels"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int

    private let _size: Size?
    public var _maxPixels: Int?
    public var _minPixels: Int?
    public var _maxFrames: Int?
    public var _fps: Double?

    public var minPixels: Int {
        get {
            _minPixels ?? _size?.minPixels ?? 3136
        }
        set {
            _minPixels = newValue
        }
    }
    public var maxPixels: Int {
        get {
            _maxPixels ?? _size?.maxPixels ?? 12_845_056
        }
        set {
            _maxPixels = newValue
        }
    }
    public var maxFrames: Int {
        get {
            _maxFrames ?? Int.max
        }
        set {
            _maxFrames = newValue
        }
    }
    public var fps: Double {
        get {
            _fps ?? 2.0
        }
        set {
            _fps = newValue
        }
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case _maxPixels = "max_pixels"
        case _minPixels = "min_pixels"
        case _maxFrames = "max_frames"
        case _fps = "fps"
        case _size = "size"
    }
}

/// Message Generator for Qwen2VL
public struct Qwen2VLMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        [
            "role": message.role.rawValue,
            "content": [
                ["type": "text", "text": message.content]
            ]
                // Messages format for Qwen 2 VL, Qwen 2.5 VL. May need to be adapted for other models.
                + message.images.map { _ in
                    ["type": "image"]
                }
                + message.videos.map { _ in
                    ["type": "video"]
                },
        ]
    }
}
