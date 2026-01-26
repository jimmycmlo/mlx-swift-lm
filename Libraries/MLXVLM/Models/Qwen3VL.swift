// Copyright © 2025 Apple Inc.

import AVFoundation
import CoreImage
import CoreMedia
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Profiling

private struct ProfilingStats {
    static var rotaryEmbeddingTime: Double = 0
    static var rotaryEmbeddingCount: Int = 0
    static var positionIDTime: Double = 0
    static var positionIDCount: Int = 0
    static var attentionTime: Double = 0
    static var attentionCount: Int = 0
    static var modelForwardTime: Double = 0
    static var modelForwardCount: Int = 0
    static var languageModelTime: Double = 0
    static var languageModelCount: Int = 0
    static var lmHeadTime: Double = 0
    static var lmHeadCount: Int = 0
    
    static func reset() {
        rotaryEmbeddingTime = 0
        rotaryEmbeddingCount = 0
        positionIDTime = 0
        positionIDCount = 0
        attentionTime = 0
        attentionCount = 0
        modelForwardTime = 0
        modelForwardCount = 0
        languageModelTime = 0
        languageModelCount = 0
        lmHeadTime = 0
        lmHeadCount = 0
    }
    
    static func printStats() {
        print("\n=== Qwen3VL Profiling Stats ===")
        if rotaryEmbeddingCount > 0 {
            let avg = rotaryEmbeddingTime * 1000 / Double(rotaryEmbeddingCount)
            print("rotaryEmbedding: \(String(format: "%.2f", rotaryEmbeddingTime * 1000))ms total, \(rotaryEmbeddingCount) calls, \(String(format: "%.2f", avg))ms avg")
        }
        if positionIDCount > 0 {
            let avg = positionIDTime * 1000 / Double(positionIDCount)
            print("positionID computation: \(String(format: "%.2f", positionIDTime * 1000))ms total, \(positionIDCount) calls, \(String(format: "%.2f", avg))ms avg")
        }
        if attentionCount > 0 {
            let avg = attentionTime * 1000 / Double(attentionCount)
            print("attention: \(String(format: "%.2f", attentionTime * 1000))ms total, \(attentionCount) calls, \(String(format: "%.2f", avg))ms avg")
        }
        if modelForwardCount > 0 {
            let avg = modelForwardTime * 1000 / Double(modelForwardCount)
            print("model forward: \(String(format: "%.2f", modelForwardTime * 1000))ms total, \(modelForwardCount) calls, \(String(format: "%.2f", avg))ms avg")
        }
        if languageModelCount > 0 {
            let avg = languageModelTime * 1000 / Double(languageModelCount)
            print("languageModel.callAsFunction: \(String(format: "%.2f", languageModelTime * 1000))ms total, \(languageModelCount) calls, \(String(format: "%.2f", avg))ms avg")
        }
        if lmHeadCount > 0 {
            let avg = lmHeadTime * 1000 / Double(lmHeadCount)
            print("lmHead: \(String(format: "%.2f", lmHeadTime * 1000))ms total, \(lmHeadCount) calls, \(String(format: "%.2f", avg))ms avg")
        }
        // Total time is languageModelTime (which includes all nested operations)
        // Don't sum nested times to avoid double-counting
        let totalTime = languageModelTime
        print("Total tracked time: \(String(format: "%.2f", totalTime * 1000))ms")
        print("==============================\n")
    }
}

// Public API for profiling
public extension Qwen3VL {
    static func printProfilingStats() {
        ProfilingStats.printStats()
    }
    
    static func resetProfilingStats() {
        ProfilingStats.reset()
    }
}

private enum Qwen3VLError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
}

// MARK: - Processor

public struct Qwen3VLProcessor: UserInputProcessor {

    public var config: Qwen3VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Qwen3VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let processed = images.map { MediaProcessing.apply($0, processing: processing) }

        guard let first = processed.first else {
            throw VLMError.imageProcessingFailure("No image provided")
        }

        let extent = first.extent.size
        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
            height: Int(extent.height),
            width: Int(extent.width),
            factor: config.patchSize * config.mergeSize,
            minPixels: config.minPixels,
            maxPixels: config.maxPixels)

        let targetSize = CGSize(width: resizedWidth, height: resizedHeight)

        let resampled = processed.map { MediaProcessing.resampleBicubic($0, to: targetSize) }

        let normalized =
            resampled
            .map {
                MediaProcessing.normalize(
                    $0,
                    mean: config.imageMeanTuple,
                    std: config.imageStdTuple)
            }
            .map { MediaProcessing.asMLXArray($0) }

        return try QwenVL.patchify(
            images: normalized,
            mergeSize: config.mergeSize,
            patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen3VLMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(messages: messages, tools: input.tools)

        if input.images.isEmpty, input.videos.isEmpty {
            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray).asType(.int8)
            return LMInput(text: .init(tokens: promptArray, mask: mask))
        }

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imageFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let concatenated = concatenated(imageFrames.map { $0.0 })
            processedImage = .init(pixels: concatenated, frames: imageFrames.map { $0.1 })

            if let frames = processedImage?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens,
                    frames: frames,
                    paddingToken: "<|image_pad|>",
                    mergeSize: config.mergeSize,
                    tokenizer: tokenizer)
            }
        }

        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var accumulatedFrames: [[MLXArray]] = []

            for video in input.videos {
                var resizedSize: CGSize = .zero
                let sequence = try await MediaProcessing.asProcessedSequence(
                    video, targetFPS: { _ in Double(config.fps) }, maxFrames: config.maxFrames
                ) { frame in
                    let processed = MediaProcessing.apply(frame.frame, processing: input.processing)
                    if resizedSize == .zero {
                        let size = processed.extent.size
                        let (height, width) = try QwenVL.targetSize(
                            height: Int(size.height),
                            width: Int(size.width),
                            factor: config.patchSize * config.mergeSize,
                            minPixels: config.minPixels,
                            maxPixels: config.maxPixels)
                        resizedSize = CGSize(width: width, height: height)
                    }
                    let finalImage = preprocess(image: processed, resizedSize: resizedSize)
                    return VideoFrame(frame: finalImage, timeStamp: frame.timeStamp)
                }
                accumulatedFrames.append(sequence.frames)
            }

            let videoFrames = try accumulatedFrames.map {
                try QwenVL.patchify(
                    images: $0,
                    mergeSize: config.mergeSize,
                    patchSize: config.patchSize,
                    temporalPatchSize: config.temporalPatchSize)
            }

            let concatenated = concatenated(videoFrames.map { $0.0 })
            processedVideo = .init(pixels: concatenated, frames: videoFrames.map { $0.1 })

            if let frames = processedVideo?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens,
                    frames: frames,
                    paddingToken: "<|video_pad|>",
                    mergeSize: config.mergeSize,
                    tokenizer: tokenizer)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo)
    }
    
    /// Prepare input with specific frame specification for selective video processing
    /// 
    /// This method allows you to control which frames from videos are processed
    /// during the preprocessing stage, enabling more efficient video analysis.
    /// 
    /// - Parameter input: The user input containing text, images, and/or videos
    /// - Parameter frameSpecification: Which frames to process from videos
    /// - Returns: The prepared LMInput with selective frame processing
    /// - Throws: VLMError if video processing fails
    public func prepareWithFrameSpecification(input: UserInput, frameSpecification: Qwen3VL.FrameSpecification) async throws -> LMInput {
        let messages = Qwen3VLMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(messages: messages, tools: input.tools)

        // Text-only input
        if input.images.isEmpty, input.videos.isEmpty {
            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray).asType(.int8)
            return LMInput(text: .init(tokens: promptArray, mask: mask))
        }

        // Process images if any
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imageFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let concatenated = concatenated(imageFrames.map { $0.0 })
            processedImage = .init(pixels: concatenated, frames: imageFrames.map { $0.1 })

            if let frames = processedImage?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens,
                    frames: frames,
                    paddingToken: "<|image_pad|>",
                    mergeSize: config.mergeSize,
                    tokenizer: tokenizer)
            }
        }

        // Process videos with frame specification
        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var accumulatedFrames: [[MLXArray]] = []
            var resizedSize: CGSize = .zero
            
            for video in input.videos {
                let imageSequence: [MLXArray]
                
                switch frameSpecification {
                case .allFrames:
                    // Process all frames as before
                    let sequence = try await MediaProcessing.asProcessedSequence(
                        video.asAVAsset(), maxFrames: config.maxFrames, targetFPS: { _ in config.fps }
                    ) { frame in
                        let processed = MediaProcessing.apply(frame.frame, processing: input.processing)
                        if resizedSize == .zero {
                            let size = processed.extent.size
                            let (height, width) = try QwenVL.targetSize(
                                height: Int(size.height),
                                width: Int(size.width),
                                factor: config.patchSize * config.mergeSize,
                                minPixels: config.minPixels,
                                maxPixels: config.maxPixels)
                            resizedSize = CGSize(width: width, height: height)
                        }
                        let finalImage = preprocess(image: processed, resizedSize: resizedSize)
                        return VideoFrame(frame: finalImage, timeStamp: frame.timeStamp)
                    }
                    imageSequence = sequence.frames
                    
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
                
                accumulatedFrames.append(imageSequence)
            }
            
            let videoFrames = try accumulatedFrames.map {
                try QwenVL.patchify(
                    images: $0,
                    mergeSize: config.mergeSize,
                    patchSize: config.patchSize,
                    temporalPatchSize: config.temporalPatchSize)
            }

            let concatenated = concatenated(videoFrames.map { $0.0 })
            processedVideo = .init(pixels: concatenated, frames: videoFrames.map { $0.1 })

            if let frames = processedVideo?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens,
                    frames: frames,
                    paddingToken: "<|video_pad|>",
                    mergeSize: config.mergeSize,
                    tokenizer: tokenizer)
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
        processing: UserInput.Processing?,
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

public struct Qwen3VLProcessorConfiguration: Codable, Sendable {

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
    private let _minPixels: Int?
    private let _maxPixels: Int?
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let imageProcessorType: String
    
    // Runtime settable properties
    public var _runtimeMaxPixels: Int?
    public var _runtimeMinPixels: Int?
    public var _maxFrames: Int?
    public var _fps: Double?

    public var minPixels: Int {
        get { _runtimeMinPixels ?? _minPixels ?? 4 * 28 * 28 }  // 3,136
        set { _runtimeMinPixels = newValue }
    }
    
    public var maxPixels: Int {
        get { _runtimeMaxPixels ?? _maxPixels ?? 16384 * 28 * 28 }  // 12,845,056
        set { _runtimeMaxPixels = newValue }
    }
    
    public var maxFrames: Int {
        get { _maxFrames ?? Int.max }
        set { _maxFrames = newValue }
    }
    
    public var fps: Double {
        get { _fps ?? 2.0 }
        set { _fps = newValue }
    }

    public var size: Size { .init(maxPixels: maxPixels, minPixels: minPixels) }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case _minPixels = "min_pixels"
        case _maxPixels = "max_pixels"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case imageProcessorType = "image_processor_type"
        case _runtimeMaxPixels = "runtime_max_pixels"
        case _runtimeMinPixels = "runtime_min_pixels"
        case _maxFrames = "max_frames"
        case _fps = "fps"
    }
}

// MARK: - Model Configuration

public struct Qwen3VLConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        private let _numKeyValueHeads: Int?
        public var numKeyValueHeads: Int { _numKeyValueHeads ?? numAttentionHeads }
        public let headDim: Int
        private let _ropeTheta: Double?
        public var ropeTheta: Double { _ropeTheta ?? 1_000_000 }
        public let maxPositionEmbeddings: Int
        private let _rmsNormEps: Double?
        public var rmsNormEps: Double { _rmsNormEps ?? 1e-6 }
        private let _ropeScaling: RoPEScaling?
        public var ropeScaling: RoPEScaling? { _ropeScaling }
        private let _normTopKProb: Bool?
        public var normTopKProb: Bool { _normTopKProb ?? true }
        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? true }
        private let _attentionBias: Bool?
        public var attentionBias: Bool { _attentionBias ?? false }
        private let _hiddenAct: String?
        public var hiddenAct: String { _hiddenAct ?? "silu" }
        public let vocabSize: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case _numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case _ropeTheta = "rope_theta"
            case maxPositionEmbeddings = "max_position_embeddings"
            case _rmsNormEps = "rms_norm_eps"
            case _ropeScaling = "rope_scaling"
            case _normTopKProb = "norm_topk_prob"
            case _tieWordEmbeddings = "tie_word_embeddings"
            case _attentionBias = "attention_bias"
            case _hiddenAct = "hidden_act"
            case vocabSize = "vocab_size"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let outHiddenSize: Int
        public let numHeads: Int
        public let patchSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        public let numPositionEmbeddings: Int
        private let _inChannels: Int?
        public var inChannels: Int { _inChannels ?? 3 }
        private let _hiddenAct: String?
        public var hiddenAct: String { _hiddenAct ?? "gelu" }
        private let _deepstackVisualIndexes: [Int]?
        public var deepstackVisualIndexes: [Int] { _deepstackVisualIndexes ?? [] }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case outHiddenSize = "out_hidden_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case numPositionEmbeddings = "num_position_embeddings"
            case _inChannels = "in_channels"
            case _hiddenAct = "hidden_act"
            case _deepstackVisualIndexes = "deepstack_visual_indexes"
        }
    }

    public struct RoPEScaling: Codable, Sendable {
        public let type: String?
        public let mropeInterleaved: Bool?
        public let mropeSection: [Int]?

        enum CodingKeys: String, CodingKey {
            case type
            case mropeInterleaved = "mrope_interleaved"
            case mropeSection = "mrope_section"
        }

        public init(type: String? = nil, mropeInterleaved: Bool? = nil, mropeSection: [Int]? = nil)
        {
            self.type = type
            self.mropeInterleaved = mropeInterleaved
            self.mropeSection = mropeSection
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _ignoreIndex: Int?
    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 151_655 }
    private let _videoTokenId: Int?
    public var videoTokenId: Int { _videoTokenId ?? 151_656 }
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    private let _visionStartTokenId: Int?
    public var visionStartTokenId: Int { _visionStartTokenId ?? 151_652 }
    private let _visionEndTokenId: Int?
    public var visionEndTokenId: Int { _visionEndTokenId ?? 151_653 }
    private let _visionTokenId: Int?
    public var visionTokenId: Int { _visionTokenId ?? 151_654 }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfiguration.vocabSize }
    private let _eosTokenId: [Int]?
    public var eosTokenId: [Int]? { _eosTokenId }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenId = "vision_start_token_id"
        case _visionEndTokenId = "vision_end_token_id"
        case _visionTokenId = "vision_token_id"
        case _vocabSize = "vocab_size"
        case _eosTokenId = "eos_token_id"
    }

    public init(
        textConfiguration: TextConfiguration, visionConfiguration: VisionConfiguration,
        modelType: String = "qwen3_vl", ignoreIndex: Int = -100, imageTokenId: Int = 151_655,
        videoTokenId: Int = 151_656, imageTokenIndex: Int? = nil, videoTokenIndex: Int? = nil,
        visionStartTokenId: Int = 151_652, visionEndTokenId: Int = 151_653,
        visionTokenId: Int = 151_654, vocabSize: Int? = nil, eosTokenId: [Int]? = nil
    ) {
        self.textConfiguration = textConfiguration
        self.visionConfiguration = visionConfiguration
        self.modelType = modelType
        self._ignoreIndex = ignoreIndex
        self._imageTokenId = imageTokenId
        self._videoTokenId = videoTokenId
        self._imageTokenIndex = imageTokenIndex
        self._videoTokenIndex = videoTokenIndex
        self._visionStartTokenId = visionStartTokenId
        self._visionEndTokenId = visionEndTokenId
        self._visionTokenId = visionTokenId
        self._vocabSize = vocabSize
        self._eosTokenId = eosTokenId
    }
}

// MARK: - Vision

enum Qwen3VLVision {

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let first = x[.ellipsis, 0 ..< half]
        let second = x[.ellipsis, half...]
        return concatenated([-second, first], axis: -1)
    }

    static func applyRotary(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cosVals = cos(freqs)
        var sinVals = sin(freqs)

        cosVals = expandedDimensions(cosVals, axis: 1)
        cosVals = tiled(cosVals, repetitions: [1, 1, 2])
        cosVals = expandedDimensions(cosVals, axis: 0)

        sinVals = expandedDimensions(sinVals, axis: 1)
        sinVals = tiled(sinVals, repetitions: [1, 1, 2])
        sinVals = expandedDimensions(sinVals, axis: 0)

        let rotated = (tensor * cosVals) + (rotateHalf(tensor) * sinVals)
        return rotated.asType(tensor.dtype)
    }

    final class VisionRotaryEmbedding {
        let dimension: Int
        let theta: Float

        init(dimension: Int, theta: Float = 10_000) {
            self.dimension = dimension
            self.theta = theta
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let invFreq =
                1.0
                / pow(
                    MLXArray(theta),
                    MLXArray(stride(from: 0, to: dimension, by: 2)).asType(.float32)
                        / Float(dimension)
                )
            let seq = MLXArray(0 ..< sequenceLength).asType(invFreq.dtype)
            return outer(seq, invFreq)
        }
    }

    final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo(key: "proj") var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let hiddenSize: Int

        init(
            patchSize: Int,
            temporalPatchSize: Int,
            inChannels: Int,
            hiddenSize: Int
        ) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.hiddenSize = hiddenSize

            let kernel = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            _proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: hiddenSize,
                kernelSize: kernel,
                stride: kernel,
                bias: true
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x.reshaped(
                -1,
                inChannels,
                temporalPatchSize,
                patchSize,
                patchSize
            ).movedAxis(source: 1, destination: 4)

            states = proj(states)
            states = states.reshaped(-1, hiddenSize)
            return states
        }
    }

    final class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        let usePostShuffleNorm: Bool

        @ModuleInfo(key: "norm") var norm: LayerNorm
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(config: Qwen3VLConfiguration.VisionConfiguration, usePostShuffleNorm: Bool) {
            self.hiddenSize =
                config.hiddenSize * (config.spatialMergeSize * config.spatialMergeSize)
            self.usePostShuffleNorm = usePostShuffleNorm

            let normDim = usePostShuffleNorm ? hiddenSize : config.hiddenSize
            _norm.wrappedValue = LayerNorm(dimensions: normDim, eps: 1e-6)
            _linear1.wrappedValue = Linear(hiddenSize, hiddenSize)
            _linear2.wrappedValue = Linear(hiddenSize, config.outHiddenSize)
            _activation.wrappedValue = GELU()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x
            if usePostShuffleNorm {
                states = states.reshaped(-1, hiddenSize)
            }
            states = norm(states)
            states = states.reshaped(-1, hiddenSize)
            states = linear1(states)
            states = activation(states)
            states = linear2(states)
            return states
        }
    }

    final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        init(dim: Int, numHeads: Int) {
            self.numHeads = numHeads
            self.headDim = dim / numHeads
            self.scale = pow(Float(headDim), -0.5)

            _qkv.wrappedValue = Linear(dim, 3 * dim, bias: true)
            _proj.wrappedValue = Linear(dim, dim)
        }

        func callAsFunction(
            _ x: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            var qkvStates = qkv(x)
            qkvStates = qkvStates.reshaped(sequenceLength, 3, numHeads, headDim)
            qkvStates = qkvStates.transposed(1, 0, 2, 3)

            let parts = split(qkvStates, parts: 3, axis: 0)
            var queries = parts[0][0, 0..., 0..., 0...]
            var keys = parts[1][0, 0..., 0..., 0...]
            var values = parts[2][0, 0..., 0..., 0...]

            queries = applyRotary(queries, freqs: rotaryPosEmb)
            keys = applyRotary(keys, freqs: rotaryPosEmb)

            queries = queries.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)

            var mask = ones([1, sequenceLength, sequenceLength], dtype: queries.dtype)
            mask = mask * MLXArray(-1e9, dtype: queries.dtype)

            let seqlens = cuSeqlens.asArray(Int.self)
            for idx in 1 ..< seqlens.count {
                let start = seqlens[idx - 1]
                let end = seqlens[idx]
                mask[0..., start ..< end, start ..< end] = MLXArray(0, dtype: queries.dtype)
            }

            let attended = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .array(mask)
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return proj(attended)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(dim: Int, hiddenDim: Int) {
            _linear1.wrappedValue = Linear(dim, hiddenDim, bias: true)
            _linear2.wrappedValue = Linear(hiddenDim, dim, bias: true)
            _activation.wrappedValue = GELU(approximation: .fast)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(activation(linear1(x)))
        }
    }

    final class VisionBlock: Module {
        @ModuleInfo(key: "norm1") var norm1: LayerNorm
        @ModuleInfo(key: "norm2") var norm2: LayerNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        init(_ config: Qwen3VLConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attention.wrappedValue = Attention(dim: config.hiddenSize, numHeads: config.numHeads)
            _mlp.wrappedValue = MLP(dim: config.hiddenSize, hiddenDim: config.intermediateSize)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray
        ) -> MLXArray {
            var states = hiddenStates
            states =
                states + attention(norm1(states), cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryPosEmb)
            states = states + mlp(norm2(states))
            return states
        }
    }

    final class VisionModel: Module {

        let config: Qwen3VLConfiguration.VisionConfiguration
        let spatialMergeSize: Int
        let numGridPerSide: Int

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryEmbedding: VisionRotaryEmbedding
        @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
        @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
        @ModuleInfo(key: "merger") var merger: PatchMerger
        @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [PatchMerger]
        let deepstackVisualIndexes: [Int]

        init(_ config: Qwen3VLConfiguration.VisionConfiguration) {
            self.config = config
            self.spatialMergeSize = config.spatialMergeSize
            self.numGridPerSide = Int(sqrt(Double(config.numPositionEmbeddings)))
            self.deepstackVisualIndexes = config.deepstackVisualIndexes

            _patchEmbed.wrappedValue = PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize)

            let headDim = config.hiddenSize / config.numHeads
            _rotaryEmbedding.wrappedValue = VisionRotaryEmbedding(dimension: headDim / 2)

            _posEmbed.wrappedValue = Embedding(
                embeddingCount: config.numPositionEmbeddings,
                dimensions: config.hiddenSize)

            _blocks.wrappedValue = (0 ..< config.depth).map { _ in VisionBlock(config) }
            _merger.wrappedValue = PatchMerger(config: config, usePostShuffleNorm: false)

            _deepstackMergers.wrappedValue = config.deepstackVisualIndexes.map { _ in
                PatchMerger(config: config, usePostShuffleNorm: true)
            }
        }

        private func rotaryPositionEmbedding(_ grids: [THW]) -> MLXArray {
            guard let maxHW = grids.map({ max($0.h, $0.w) }).max(), maxHW > 0 else {
                return MLXArray.zeros([1, 1], dtype: .float32)
            }

            let freqTable = rotaryEmbedding(sequenceLength: maxHW)
            let halfDim = freqTable.dim(-1)

            let merge = spatialMergeSize
            var allCoords: [MLXArray] = []
            let mergeScalar = MLXArray(Int32(merge))

            for grid in grids {
                let mergedH = grid.h / merge
                let mergedW = grid.w / merge

                guard mergedH > 0, mergedW > 0 else { continue }

                // Generate block and intra-block indices fully in MLX
                var blockRows = MLXArray(0 ..< mergedH).asType(.int32)
                blockRows = blockRows.reshaped([mergedH, 1, 1, 1])

                var blockCols = MLXArray(0 ..< mergedW).asType(.int32)
                blockCols = blockCols.reshaped([1, mergedW, 1, 1])

                let intra = MLXArray(0 ..< merge).asType(.int32)
                let intraRow = intra.reshaped([1, 1, merge, 1])
                let intraCol = intra.reshaped([1, 1, 1, merge])

                // Broadcast arithmetic mirrors the Python implementation
                var hIndex = blockRows * mergeScalar + intraRow
                var wIndex = blockCols * mergeScalar + intraCol

                hIndex = broadcast(hIndex, to: [mergedH, mergedW, merge, merge])
                wIndex = broadcast(wIndex, to: [mergedH, mergedW, merge, merge])

                // Flatten and stack coordinate pairs
                let hFlattened = hIndex.flattened()
                let wFlattened = wIndex.flattened()
                var coords = stacked([hFlattened, wFlattened], axis: -1)

                // Repeat for temporal frames
                if grid.t > 1 {
                    coords = tiled(coords, repetitions: [grid.t, 1])
                }

                allCoords.append(coords)
            }

            guard !allCoords.isEmpty else {
                return MLXArray.zeros([0, halfDim * 2], dtype: freqTable.dtype)
            }

            // Concatenate all coordinate pairs
            let allCoordsConcat = concatenated(allCoords, axis: 0)  // (total_tokens, 2)

            // Extract h and w indices and lookup embeddings
            let hIndices = allCoordsConcat[0..., 0].asType(.int32)
            let wIndices = allCoordsConcat[0..., 1].asType(.int32)

            let hEmbeds = freqTable[hIndices, 0...]
            let wEmbeds = freqTable[wIndices, 0...]

            // Concatenate height and width embeddings
            return concatenated([hEmbeds, wEmbeds], axis: -1)
        }

        private func positionalEmbeddings(_ grids: [THW]) -> MLXArray {
            let hiddenSize = config.hiddenSize
            let maxIndex = numGridPerSide - 1

            // Step 1: Collect all indices and weights from all grids using MLX ops
            var cornerIndices: [[MLXArray]] = Array(repeating: [], count: 4)
            var cornerWeights: [[MLXArray]] = Array(repeating: [], count: 4)
            var gridSizes: [Int] = []

            for grid in grids {
                let h = grid.h
                let w = grid.w
                gridSizes.append(h * w)

                // Create linspace indices using broadcasting
                var hLinspace = MLXArray(0 ..< h).asType(.float32)
                hLinspace = hLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, h - 1)))

                var wLinspace = MLXArray(0 ..< w).asType(.float32)
                wLinspace = wLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, w - 1)))

                // Get floor/ceil and deltas
                let hFloor = hLinspace.asType(.int32)
                let hCeil = minimum(hFloor + 1, maxIndex)
                let dh = hLinspace - hFloor.asType(.float32)

                let wFloor = wLinspace.asType(.int32)
                let wCeil = minimum(wFloor + 1, maxIndex)
                let dw = wLinspace - wFloor.asType(.float32)

                // Broadcast to create meshgrid
                let hFloorExpanded = expandedDimensions(hFloor, axis: 1)  // (h, 1)
                let hCeilExpanded = expandedDimensions(hCeil, axis: 1)
                let wFloorExpanded = expandedDimensions(wFloor, axis: 0)  // (1, w)
                let wCeilExpanded = expandedDimensions(wCeil, axis: 0)

                let baseH = hFloorExpanded * numGridPerSide
                let baseHCeil = hCeilExpanded * numGridPerSide

                // Compute 4 corner indices
                cornerIndices[0].append((baseH + wFloorExpanded).flattened())
                cornerIndices[1].append((baseH + wCeilExpanded).flattened())
                cornerIndices[2].append((baseHCeil + wFloorExpanded).flattened())
                cornerIndices[3].append((baseHCeil + wCeilExpanded).flattened())

                // Compute bilinear weights
                let dhExpanded = expandedDimensions(dh, axis: 1)
                let dwExpanded = expandedDimensions(dw, axis: 0)

                cornerWeights[0].append(((1 - dhExpanded) * (1 - dwExpanded)).flattened())
                cornerWeights[1].append(((1 - dhExpanded) * dwExpanded).flattened())
                cornerWeights[2].append((dhExpanded * (1 - dwExpanded)).flattened())
                cornerWeights[3].append((dhExpanded * dwExpanded).flattened())
            }

            guard !cornerIndices[0].isEmpty else {
                return MLXArray.zeros([0, hiddenSize], dtype: posEmbed.weight.dtype)
            }

            // Step 2: Batch embedding lookup
            let indicesTensors = cornerIndices.map { concatenated($0, axis: 0).asType(.int32) }
            let weightsTensors = cornerWeights.map {
                concatenated($0, axis: 0).asType(posEmbed.weight.dtype)
            }

            let totalPatches = indicesTensors[0].dim(0)
            var patchPosEmbeds = MLXArray.zeros(
                [totalPatches, hiddenSize], dtype: posEmbed.weight.dtype)

            for cornerIdx in 0 ..< 4 {
                let cornerEmbeds = posEmbed(indicesTensors[cornerIdx])
                let weighted =
                    cornerEmbeds * expandedDimensions(weightsTensors[cornerIdx], axis: -1)
                patchPosEmbeds = patchPosEmbeds + weighted
            }

            // Step 3: Split by grid (like Python lines 344-349)
            var patchPosEmbedsSplit: [MLXArray] = []
            var offset = 0

            for size in gridSizes {
                let slice = patchPosEmbeds[offset ..< (offset + size), 0...]
                patchPosEmbedsSplit.append(slice)
                offset += size
            }

            // Step 4: Process each grid (like Python lines 354-371)
            var resultEmbeds: [MLXArray] = []
            let merge = spatialMergeSize

            for (gridIdx, grid) in grids.enumerated() {
                let posEmbed = patchPosEmbedsSplit[gridIdx]
                let h = grid.h
                let w = grid.w
                let t = grid.t

                let featureDim = posEmbed.dim(-1)

                // Repeat for temporal dimension
                var temporalEmbeds = tiled(posEmbed, repetitions: [t, 1])

                // Reshape for merge pattern
                temporalEmbeds = temporalEmbeds.reshaped(t, h, w, featureDim)
                temporalEmbeds = temporalEmbeds.reshaped(
                    t,
                    h / merge,
                    merge,
                    w / merge,
                    merge,
                    featureDim
                )
                temporalEmbeds = temporalEmbeds.transposed(0, 1, 3, 2, 4, 5)
                temporalEmbeds = temporalEmbeds.reshaped(-1, featureDim)

                resultEmbeds.append(temporalEmbeds)
            }

            return concatenated(resultEmbeds, axis: 0)
        }

        private func cumulativeSequenceLengths(_ grids: [THW]) -> MLXArray {
            var seqLengths: [MLXArray] = []

            for grid in grids {
                let perFrame = grid.h * grid.w
                let repeated = tiled(MLXArray(perFrame), repetitions: [grid.t])
                seqLengths.append(repeated)
            }

            guard !seqLengths.isEmpty else {
                return MLXArray(0, dtype: .int32)
            }

            let concatSeqLengths = concatenated(seqLengths).asType(.int32)

            let cumSum = concatSeqLengths.cumsum()

            return padded(
                cumSum, widths: [IntOrPair((1, 0))], mode: .constant,
                value: MLXArray(0, dtype: cumSum.dtype))
        }

        func callAsFunction(_ pixelValues: MLXArray, gridTHW: [THW]) -> (MLXArray, [MLXArray]) {
            var hiddenStates = patchEmbed(pixelValues)

            let posEmbeds = positionalEmbeddings(gridTHW)
            hiddenStates = hiddenStates + posEmbeds

            let rotaryEmbeds = rotaryPositionEmbedding(gridTHW)
            let cuSeqlens = cumulativeSequenceLengths(gridTHW)

            var deepstackOutputs: [MLXArray] = []

            for (index, block) in blocks.enumerated() {
                hiddenStates = block(hiddenStates, cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryEmbeds)
                if let dsIndex = deepstackVisualIndexes.firstIndex(of: index) {
                    let feature = deepstackMergers[dsIndex](hiddenStates)
                    deepstackOutputs.append(feature)
                }
            }

            hiddenStates = merger(hiddenStates)
            return (hiddenStates, deepstackOutputs)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                } else if key.contains("patch_embed.proj.weight") {
                    if value.ndim == 5 && value.dim(-1) == config.inChannels {
                        sanitized[key] = value
                    } else {
                        sanitized[key] = value.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}

// MARK: - Language

enum Qwen3VLLanguage {

    final class RotaryEmbedding {

        private let invFreq: MLXArray
        private let mropeSection: [Int]
        private var cachedHMask: MLXArray?
        private var cachedWMask: MLXArray?
        private var cachedDims: Int = 0
        private var cachedExpandedHMask: MLXArray?
        private var cachedExpandedWMask: MLXArray?
        private var cachedBatch: Int = 0
        private var cachedSeqLen: Int = 0

        init(headDim: Int, base: Double, ropeScaling: Qwen3VLConfiguration.RoPEScaling?) {
            var freq = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
            freq = freq / Float(headDim)
            let baseArray = MLXArray(Float(base))
            self.invFreq = 1.0 / pow(baseArray, freq)
            self.mropeSection = ropeScaling?.mropeSection ?? [24, 20, 20]
        }
        
        private func getMasks(dims: Int, batch: Int, seqLen: Int) -> (MLXArray?, MLXArray?) {
            // Cache base masks if dimensions haven't changed
            if cachedDims != dims || cachedHMask == nil || cachedWMask == nil {
                let hEnd = min(mropeSection[1] * 3, dims)
                let wEnd = min(mropeSection[2] * 3, dims)
                
                let indices = MLXArray(0 ..< dims).asType(.int32)
                
                if hEnd > 1 {
                    let hMask = (indices .>= 1) .&& (indices .< hEnd) .&& ((indices - 1) % 3 .== 0)
                    cachedHMask = hMask
                } else {
                    cachedHMask = nil
                }
                
                if wEnd > 2 {
                    let wMask = (indices .>= 2) .&& (indices .< wEnd) .&& ((indices - 2) % 3 .== 0)
                    cachedWMask = wMask
                } else {
                    cachedWMask = nil
                }
                
                cachedDims = dims
                // Invalidate expanded masks when base masks change
                cachedExpandedHMask = nil
                cachedExpandedWMask = nil
            }
            
            // Cache expanded masks if batch/seqLen haven't changed (common during generation)
            if cachedBatch != batch || cachedSeqLen != seqLen || cachedExpandedHMask == nil || cachedExpandedWMask == nil {
                if let hMask = cachedHMask {
                    cachedExpandedHMask = broadcast(hMask.reshaped(1, 1, dims), to: [batch, seqLen, dims])
                } else {
                    cachedExpandedHMask = nil
                }
                
                if let wMask = cachedWMask {
                    cachedExpandedWMask = broadcast(wMask.reshaped(1, 1, dims), to: [batch, seqLen, dims])
                } else {
                    cachedExpandedWMask = nil
                }
                
                cachedBatch = batch
                cachedSeqLen = seqLen
            }
            
            return (cachedExpandedHMask, cachedExpandedWMask)
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            // Python: freqs_t = freqs[0]; freqs_t[..., idx] = freqs[dim, ..., idx]
            // Use cached masks to avoid recomputing them every time
            var freqs_t = freqs[0, 0..., 0..., 0...]  // (bs, seq_len, head_dim // 2)

            let dims = freqs_t.dim(-1)
            let batch = freqs_t.dim(0)
            let seqLen = freqs_t.dim(1)
            
            // Get cached masks (only recompute if dimensions changed)
            let (hMaskExpanded, wMaskExpanded) = getMasks(dims: dims, batch: batch, seqLen: seqLen)
            
            // Apply h mask if present
            if let hMaskExpanded = hMaskExpanded {
                let freqs_h = freqs[1, 0..., 0..., 0...]
                freqs_t = `where`(hMaskExpanded, freqs_h, freqs_t)
            }
            
            // Apply w mask if present
            if let wMaskExpanded = wMaskExpanded {
                let freqs_w = freqs[2, 0..., 0..., 0...]
                freqs_t = `where`(wMaskExpanded, freqs_w, freqs_t)
            }

            return freqs_t
        }

        func callAsFunction(positionIds: MLXArray, dtype: MLX.DType) -> (MLXArray, MLXArray) {
            let startTime = CFAbsoluteTimeGetCurrent()
            defer {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                ProfilingStats.rotaryEmbeddingTime += elapsed
                ProfilingStats.rotaryEmbeddingCount += 1
            }
            
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = positionIds[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds, repetitions: [3, 1, 1])
            }

            let pos = positionIds.asType(.float32)
            var invFreq = self.invFreq.asType(.float32)
            invFreq = invFreq[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * invFreq
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            let cosValues = cos(emb).asType(dtype)
            let sinValues = sin(emb).asType(dtype)
            return (cosValues, sinValues)
        }
    }

    static func applyMultimodalRotary(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        var cos = cos
        var sin = sin
        cos = expandedDimensions(cos, axis: 1)
        sin = expandedDimensions(sin, axis: 1)
        let qEmbedded = (q * cos) + (QwenVL.rotateHalf(q) * sin)
        let kEmbedded = (k * cos) + (QwenVL.rotateHalf(k) * sin)
        return (qEmbedded, kEmbedded)
    }

    final class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.numAttentionHeads
            self.kvHeads = config.numKeyValueHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

            rotaryEmbedding = RotaryEmbedding(
                headDim: headDim,
                base: config.ropeTheta,
                ropeScaling: config.ropeScaling)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let startTime = CFAbsoluteTimeGetCurrent()
            defer {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                ProfilingStats.attentionTime += elapsed
                ProfilingStats.attentionCount += 1
            }
            
            let (batch, length) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(batch, length, heads, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)

            keys = keys.reshaped(batch, length, kvHeads, headDim)
            keys = kNorm(keys).transposed(0, 2, 1, 3)

            values = values.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

            var kvSequenceLength = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSequenceLength += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + length, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else {
                if let cache {
                    kvSequenceLength += cache.offset + 1
                }
            }

            let (cosValues, sinValues) = rotaryEmbedding(positionIds: positionIds!, dtype: x.dtype)

            (queries, keys) = Qwen3VLLanguage.applyMultimodalRotary(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                let slicedMask = mask[.ellipsis, 0 ..< kvSequenceLength]
                attentionMask = .array(slicedMask)
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            let result = wo(output)

            return result
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class DecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            var residual = attention(
                inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
            let hidden = x + residual
            residual = mlp(postAttentionLayerNorm(hidden))
            return hidden + residual
        }
    }

    final class Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            precondition(config.vocabSize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize)
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in DecoderLayer(config) }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?
        ) -> MLXArray {
            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let inputIds {
                hidden = embedTokens(inputIds)
            } else {
                fatalError("Either input ids or embeddings must be provided")
            }

            var mask = mask
            if mask == nil {
                mask = createAttentionMask(h: hidden, cache: cache)
            }

            for (index, layer) in layers.enumerated() {
                let layerCache = cache?[index]
                hidden = layer(hidden, mask: mask, cache: layerCache, positionIds: positionIds)

                if let embeds = deepstackEmbeds, index < embeds.count,
                    let visualMask
                {

                    hidden = applyDeepstack(
                        hiddenStates: hidden,
                        visualMask: visualMask,
                        visualEmbeds: embeds[index])
                }
            }

            return norm(hidden)
        }

        private func applyDeepstack(
            hiddenStates: MLXArray,
            visualMask: MLXArray,
            visualEmbeds: MLXArray
        ) -> MLXArray {
            let indices = maskIndices(visualMask)
            guard !indices.isEmpty else { return hiddenStates }

            let indexArray = MLXArray(indices.map { UInt32($0) })

            let result = hiddenStates
            result[0..., indexArray, 0...] = result[0..., indexArray, 0...] + visualEmbeds

            return result
        }

        private func maskIndices(_ mask: MLXArray) -> [Int] {
            let bools = mask.asType(.bool).asArray(Bool.self)
            var indices: [Int] = []
            indices.reserveCapacity(bools.count)
            for (idx, value) in bools.enumerated() where value {
                indices.append(idx)
            }
            return indices
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {

        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen3VLConfiguration
        let textConfig: Qwen3VLConfiguration.TextConfiguration
        var kvHeads: [Int]

        private var ropeDeltas: MLXArray? = nil

        init(_ config: Qwen3VLConfiguration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(
                repeating: config.textConfiguration.numKeyValueHeads,
                count: config.textConfiguration.numHiddenLayers)

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabSize,
                    bias: false)
            }
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds providedPositionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?,
            pixelValues: MLXArray?,
            imageGridTHW: [THW]?,
            videoGridTHW: [THW]?
        ) -> LMOutput {
            let startTime = CFAbsoluteTimeGetCurrent()
            defer {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                ProfilingStats.languageModelTime += elapsed
                ProfilingStats.languageModelCount += 1
            }
            
            if pixelValues != nil {
                ropeDeltas = nil
            }

            var positionIds = providedPositionIds

            if positionIds == nil && (mask == nil || mask?.ndim == 2) {
                let posIDStartTime = CFAbsoluteTimeGetCurrent()
                defer {
                    let elapsed = CFAbsoluteTimeGetCurrent() - posIDStartTime
                    ProfilingStats.positionIDTime += elapsed
                    ProfilingStats.positionIDCount += 1
                }
                
                if (cache?.first?.offset ?? 0) == 0 || ropeDeltas == nil || cache == nil {
                    if let inputIds {
                        let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                            inputIds: inputIds,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenIndex,
                            videoTokenId: config.videoTokenIndex,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: mask)

                        positionIds = computed
                        ropeDeltas = deltas
                    } else if let cache, ropeDeltas == nil {
                        let batch = inputEmbeddings!.dim(0)
                        let seqLength = inputEmbeddings!.dim(1)
                        let currentOffset = cache.first?.offset ?? 0

                        var base = MLXArray(0 ..< seqLength).asType(.int32)
                        base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                        let offsetValue = MLXArray(currentOffset).asType(.int32)
                        base = base + offsetValue

                        positionIds = base[.newAxis, 0..., 0...]
                        positionIds = tiled(positionIds!, repetitions: [3, batch, seqLength])
                    }
                } else if let cache, let ropeDeltas {
                    let batch = (inputIds ?? inputEmbeddings!).dim(0)
                    let seqLength = (inputIds ?? inputEmbeddings!).dim(1)

                    let lastCacheOffset = cache.last?.offset ?? 0

                    var delta = MLXArray(lastCacheOffset).asType(.int32) + ropeDeltas.asType(.int32)

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = base[.newAxis, 0...]
                    base = broadcast(base, to: [batch, seqLength])

                    if delta.dim(0) == 1 && batch > 1 {
                        delta = repeated(delta, count: batch, axis: 0)
                    }

                    base = base + delta

                    positionIds = base[.newAxis, 0..., 0...]
                    positionIds = broadcast(positionIds!, to: [3, batch, seqLength])
                }
            }

            let modelStartTime = CFAbsoluteTimeGetCurrent()
            defer {
                let elapsed = CFAbsoluteTimeGetCurrent() - modelStartTime
                ProfilingStats.modelForwardTime += elapsed
                ProfilingStats.modelForwardCount += 1
            }
            
            var output = model(
                inputIds,
                cache: cache,
                inputEmbeddings: inputEmbeddings,
                mask: nil,
                positionIds: positionIds,
                visualMask: visualMask,
                deepstackEmbeds: deepstackEmbeds)

            let lmHeadStartTime = CFAbsoluteTimeGetCurrent()
            if let lmHead {
                output = lmHead(output)
            } else {
                output = model.embedTokens.asLinear(output)
            }
            let lmHeadElapsed = CFAbsoluteTimeGetCurrent() - lmHeadStartTime
            ProfilingStats.lmHeadTime += lmHeadElapsed
            ProfilingStats.lmHeadCount += 1

            return LMOutput(logits: output)
        }

    }
}

extension Qwen3VLLanguage {

    static func getRopeIndex(
        inputIds: MLXArray,
        imageGridTHW: [THW]?,
        videoGridTHW: [THW]?,
        spatialMergeSize: Int,
        imageTokenId: Int,
        videoTokenId: Int,
        visionStartTokenId: Int,
        attentionMask: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {

        let (batchSize, seqLength) = (inputIds.dim(0), inputIds.dim(1))

        var positionIds = MLXArray(0 ..< seqLength).asType(.int32)
        positionIds = broadcast(positionIds[.newAxis, 0...], to: [batchSize, seqLength])

        guard inputIds.ndim > 0, imageGridTHW != nil || videoGridTHW != nil else {
            let positionIds3D = broadcast(
                positionIds[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
            let zeros = MLXArray.zeros([batchSize], dtype: .int32)
            return (positionIds3D, zeros)
        }

        positionIds = ones(like: inputIds).asType(.int32)
        positionIds = broadcast(positionIds[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])

        var mropePositionDeltas: [Int] = []
        let mask = attentionMask ?? ones(like: inputIds)

        // Process each batch item (assume batch=1 for now)
        for batchIdx in 0 ..< batchSize {
            var batchInputIds = inputIds[batchIdx, 0...]

            // Mask out padding - use where from MLX module
            batchInputIds = `where`(
                mask[batchIdx, 0...] .== 1, batchInputIds, zeros(like: batchInputIds))

            // Count images and videos in this sequence
            let visionStartMask = (batchInputIds .== MLXArray(visionStartTokenId))
            let visionStartWeighted = `where`(
                visionStartMask, MLXArray(0 ..< seqLength), zeros(like: batchInputIds))
            let visionStartIdx = argMax(visionStartWeighted).item(Int.self)

            guard visionStartIdx < seqLength - 1 else {
                continue  // No vision tokens
            }

            let imageNums = ((batchInputIds .== MLXArray(imageTokenId)).asType(.int32).sum()).item(
                Int.self)
            let videoNums = ((batchInputIds .== MLXArray(videoTokenId)).asType(.int32).sum()).item(
                Int.self)

            let inputTokens = batchInputIds.asArray(Int32.self).map { Int($0) }
            var llmPosIdsList: [MLXArray] = []

            var st = 0
            var remainImages = imageNums
            var remainVideos = videoNums
            var imageIndex = 0
            var videoIndex = 0

            // Process each image/video in sequence
            for _ in 0 ..< (imageNums + videoNums) {
                // Find next image/video token position
                let edImage: Int
                if remainImages > 0, let idx = inputTokens[st...].firstIndex(of: imageTokenId) {
                    edImage = idx
                } else {
                    edImage = inputTokens.count + 1
                }

                let edVideo: Int
                if remainVideos > 0, let idx = inputTokens[st...].firstIndex(of: videoTokenId) {
                    edVideo = idx
                } else {
                    edVideo = inputTokens.count + 1
                }

                let (t, h, w, ed): (Int, Int, Int, Int)
                if edImage < edVideo {
                    // Process image
                    guard let grid = imageGridTHW, imageIndex < grid.count else { break }
                    (t, h, w) = grid[imageIndex].values
                    imageIndex += 1
                    remainImages -= 1
                    ed = edImage
                } else {
                    // Process video
                    guard let grid = videoGridTHW, videoIndex < grid.count else { break }
                    (t, h, w) = grid[videoIndex].values
                    videoIndex += 1
                    remainVideos -= 1
                    ed = edVideo
                }

                let llmGridT = t
                let llmGridH = h / spatialMergeSize
                let llmGridW = w / spatialMergeSize

                // Calculate starting index
                let stIdx: Int
                if let lastArray = llmPosIdsList.last {
                    let maxVal = lastArray.max().item(Int.self)
                    stIdx = maxVal + 1
                } else {
                    stIdx = 0
                }

                // Add text tokens before this visual block
                let textLen = ed - st
                if textLen > 0 {
                    var index = MLXArray(0 ..< textLen).reshaped([1, textLen])
                    index = broadcast(index, to: [3, textLen])
                    index = index + MLXArray(stIdx)
                    llmPosIdsList.append(index)
                }

                // Add 3D position IDs for visual tokens
                // Python: mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                // Adds offset to ALL three dimensions!
                var tIndex = MLXArray(0 ..< llmGridT).reshaped([llmGridT, 1])
                tIndex = broadcast(tIndex, to: [llmGridT, llmGridH * llmGridW])
                tIndex = tIndex.flattened()

                var hIndex = MLXArray(0 ..< llmGridH).reshaped([1, llmGridH, 1])
                hIndex = broadcast(hIndex, to: [llmGridT, llmGridH, llmGridW])
                hIndex = hIndex.flattened()

                var wIndex = MLXArray(0 ..< llmGridW).reshaped([1, 1, llmGridW])
                wIndex = broadcast(wIndex, to: [llmGridT, llmGridH, llmGridW])
                wIndex = wIndex.flattened()

                let visualPosIds = stacked([tIndex, hIndex, wIndex]) + MLXArray(textLen + stIdx)
                llmPosIdsList.append(visualPosIds)

                st = ed + llmGridT * llmGridH * llmGridW
            }

            // Add remaining text tokens after last visual block
            if st < inputTokens.count {
                let stIdx: Int
                if let lastArray = llmPosIdsList.last {
                    let maxVal = lastArray.max().item(Int.self)
                    stIdx = maxVal + 1
                } else {
                    stIdx = 0
                }

                let textLen = inputTokens.count - st
                var tIndex = MLXArray(0 ..< textLen).reshaped([1, textLen])
                tIndex = broadcast(tIndex, to: [3, textLen])
                llmPosIdsList.append(tIndex + MLXArray(stIdx))
            }

            // Concatenate all position IDs for this batch item
            if !llmPosIdsList.isEmpty {
                let llmPositions = concatenated(llmPosIdsList, axis: 1)  // [3, seq]

                // Update position_ids for this batch
                let expandedMask = broadcast(
                    mask[batchIdx, 0...][.newAxis, .newAxis, 0...], to: [3, 1, seqLength])
                let expandedPositions = llmPositions[0..., .newAxis, 0...]
                let newPositions = `where`(
                    expandedMask, expandedPositions,
                    positionIds[0..., batchIdx ..< batchIdx + 1, 0...])

                // Replace this batch's position IDs (assumes batch size = 1)
                positionIds = newPositions

                let maxPosId = llmPositions.max().item(Int.self)
                mropePositionDeltas.append(maxPosId + 1 - inputTokens.count)
            }
        }

        // Python always returns deltas array (zeros for text-only, computed values for multimodal)
        let deltas: MLXArray
        if mropePositionDeltas.isEmpty {
            deltas = MLXArray.zeros([batchSize], dtype: .int32)
        } else {
            deltas = MLXArray(mropePositionDeltas.map { Int32($0) })
        }
        return (positionIds, deltas)
    }
}

// MARK: - Model

public final class Qwen3VL: Module, VLMModel, KVCacheDimensionProvider {

    /// Frame specification for selective video processing
    public enum FrameSpecification {
        /// Process specific frame numbers (0-based indexing)
        case frameNumbers([Int])
        /// Process frames at specific timestamps (in seconds)
        case timestamps([TimeInterval])
        /// Process all frames (default behavior)
        case allFrames
    }

    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Qwen3VLLanguage.LanguageModel

    public let config: Qwen3VLConfiguration

    public init(_ config: Qwen3VLConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen3VLLanguage.LanguageModel(config)
    }

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        var specialMask = (imageMask .|| videoMask)

        let nImageTokens = specialMask.sum().item(Int.self)

        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)

        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size

        guard nImageMaskElements == imageFeatureSize else {
            throw Qwen3VLError.featureTokenMismatch(expected: nImageTokens, actual: nImageFeatures)
        }

        let originalShape = inputEmbeds.shape
        let flattenedEmbeds = inputEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened()

        let indices = nonZero(flattenedMask.asType(.bool))

        var result = flattenedEmbeds
        if !indices.isEmpty && indices.count == flattenedFeatures.size {
            let indexArray = MLXArray(indices.map { UInt32($0) })
            result[indexArray] = flattenedFeatures
        }

        result = result.reshaped(originalShape)

        let visualMask = specialMask.squeezed(axis: -1).asType(.bool)
        return (result, visualMask)
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func combinedFrames(
        imageFrames: [THW]?,
        videoFrames: [THW]?
    ) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    private func cumulativeSplitIndices(from sizes: [Int]) -> [Int] {
        var sum = 0
        return sizes.dropLast().map { size in
            sum += size
            return sum
        }
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens

        var pixelValues: MLXArray?
        var imageFrames: [THW]? = nil
        var videoFrames: [THW]? = nil

        let dtype = visionModel.patchEmbed.proj.weight.dtype

        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(dtype))
            imageFrames = image.frames
        }

        if let video = input.video {
            pixelParts.append(video.pixels.asType(dtype))
            videoFrames = video.frames
        }

        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
        }

        var inputEmbeddings: MLXArray? = nil
        var visualMask: MLXArray?
        var deepstackEmbeds: [MLXArray]? = nil

        if let pixelValues,
            let framesList = combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames)
                .nilIfEmpty
        {
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let (visionHidden, deepstackOutputs) = visionModel(pixelValues, gridTHW: framesList)
            let mergeSize = config.visionConfiguration.spatialMergeSize
            let splits = framesList.map { $0.product / (mergeSize * mergeSize) }
            let splitIndices = cumulativeSplitIndices(from: splits)
            let featureSlices = visionHidden.split(indices: splitIndices)
            let flattenedFeatures = concatenated(featureSlices).asType(textEmbeds.dtype)

            let (mergedEmbeds, mask) = try mergeInputIdsWithImageFeatures(
                imageFeatures: flattenedFeatures,
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex)

            inputEmbeddings = mergedEmbeds
            visualMask = mask

            if !deepstackOutputs.isEmpty {
                deepstackEmbeds = deepstackOutputs.map { layerFeatures in
                    let splitIndices = cumulativeSplitIndices(from: splits)
                    let slices = layerFeatures.split(indices: splitIndices)
                    let concatenatedSlices = concatenated(slices).asType(textEmbeds.dtype)
                    return concatenatedSlices
                }
            }
        }

        let typedCache = castCache(cache)

        let languageOutput = languageModel(
            inputIds,
            cache: typedCache,
            inputEmbeddings: inputEmbeddings,
            mask: nil,
            positionIds: nil,
            visualMask: visualMask,
            deepstackEmbeds: deepstackEmbeds,
            pixelValues: pixelValues,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames)

        return .logits(languageOutput)
    }
    
    /// Prepare input with frame specification for selective video processing
    /// 
    /// This method allows you to control which frames from videos are processed
    /// during the preprocessing stage, enabling more efficient video analysis.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen3VL(config)
    /// let result = try model.prepareWithFrameSpecification(
    ///     input: lmInput, 
    ///     cache: cache, 
    ///     windowSize: windowSize,
    ///     frameSpecification: .frameNumbers([0, 10, 20, 30])
    /// )
    /// ```
    /// 
    /// - Parameter input: The LMInput containing processed text, images, and/or videos
    /// - Parameter cache: The KV cache for the model
    /// - Parameter windowSize: Optional window size for processing
    /// - Parameter frameSpecification: Which frames to process (frame numbers, timestamps, or all frames)
    /// - Parameter fps: Frames per second for timestamp conversion (default: 2.0)
    /// - Returns: The prepared result with selective frame processing
    /// - Throws: VLMError if video processing fails
    public func prepareWithFrameSpecification(_ input: LMInput, cache: [any KVCache], windowSize _: Int?, frameSpecification: FrameSpecification, fps: Double = 2.0) throws -> PrepareResult {
        let inputIds = input.text.tokens
        let inputMask = input.text.mask

        var pixelValues: MLXArray?
        var imageFrames: [THW]? = nil
        var videoFrames: [THW]? = nil

        let dtype = visionModel.patchEmbed.proj.weight.dtype

        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(dtype))
            imageFrames = image.frames
        }

        if let video = input.video, let frames = video.frames {
            // Apply frame specification filtering to video frames
            let (filteredPixels, filteredFrames) = try applyFrameSpecificationToVideo(
                videoPixels: video.pixels,
                videoFrames: frames,
                frameSpecification: frameSpecification,
                fps: fps
            )
            pixelParts.append(filteredPixels.asType(dtype))
            videoFrames = filteredFrames
        }

        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
        }

        var inputEmbeddings: MLXArray? = nil
        var visualMask: MLXArray?
        var deepstackEmbeds: [MLXArray]? = nil

        if let pixelValues,
            let framesList = combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames)
                .nilIfEmpty
        {
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let (visionHidden, deepstackOutputs) = visionModel(pixelValues, gridTHW: framesList)
            let mergeSize = config.visionConfiguration.spatialMergeSize
            let splits = framesList.map { $0.product / (mergeSize * mergeSize) }
            let splitIndices = cumulativeSplitIndices(from: splits)
            let featureSlices = visionHidden.split(indices: splitIndices)
            let flattenedFeatures = concatenated(featureSlices).asType(textEmbeds.dtype)

            let (mergedEmbeds, mask) = try mergeInputIdsWithImageFeatures(
                imageFeatures: flattenedFeatures,
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex)

            inputEmbeddings = mergedEmbeds
            visualMask = mask

            if !deepstackOutputs.isEmpty {
                deepstackEmbeds = deepstackOutputs.map { layerFeatures in
                    let splitIndices = cumulativeSplitIndices(from: splits)
                    let slices = layerFeatures.split(indices: splitIndices)
                    let concatenatedSlices = concatenated(slices).asType(textEmbeds.dtype)
                    return concatenatedSlices
                }
            }
        }

        let typedCache = castCache(cache)

        let languageOutput = languageModel(
            inputIds,
            cache: typedCache,
            inputEmbeddings: inputEmbeddings,
            mask: nil,
            positionIds: nil,
            visualMask: visualMask,
            deepstackEmbeds: deepstackEmbeds,
            pixelValues: pixelValues,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames)

        return .logits(languageOutput)
    }
    
    /// Helper method to apply frame specification filtering to video data
    private func applyFrameSpecificationToVideo(
        videoPixels: MLXArray, 
        videoFrames: [THW],
        frameSpecification: FrameSpecification,
        fps: Double
    ) throws -> (MLXArray, [THW]) {
        switch frameSpecification {
        case .allFrames:
            return (videoPixels, videoFrames)
            
        case .frameNumbers(let frameNumbers):
            let validIndices = frameNumbers.filter { $0 >= 0 && $0 < videoFrames.count }
            if validIndices.isEmpty {
                throw NSError(domain: "VideoProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "No valid frame numbers provided"])
            }
            
            let filteredPixels = videoPixels[MLXArray(validIndices), 0..., 0..., 0...]
            let filteredFrames = validIndices.map { videoFrames[$0] }
            return (filteredPixels, filteredFrames)
            
        case .timestamps(let timestamps):
            // For timestamps, we assume the frames are evenly distributed
            // This is a simplified implementation - in practice, you'd want to map timestamps to actual frame indices
            let frameNumbers = timestamps.map { Int($0 * fps) } // Use provided FPS for timestamp conversion
            return try applyFrameSpecificationToVideo(
                videoPixels: videoPixels, 
                videoFrames: videoFrames,
                frameSpecification: .frameNumbers(frameNumbers),
                fps: fps
            )
        }
    }

    /// Extract patch embeddings from a single image
    /// 
    /// This function preprocesses the image and applies patch embedding to get the initial hidden states.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen3VL(config)
    /// let processorConfig = Qwen3VLProcessorConfiguration(...)
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
        processorConfig: Qwen3VLProcessorConfiguration
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
        print("patchSize: \(processorConfig.patchSize), mergeSize: \(processorConfig.mergeSize), minPixels: \(processorConfig.minPixels), maxPixels: \(processorConfig.maxPixels)")

        // Preprocess the image (resize, normalize)
        let normalizedImage = processedImage
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: processorConfig.imageMeanTuple, std: processorConfig.imageStdTuple)
        
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

    /// Extract patch embeddings from a video by processing each frame
    /// 
    /// This function processes each frame of a video, extracts patch embeddings,
    /// and returns an array of patch embeddings for each frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen3VL(config)
    /// let processorConfig = Qwen3VLProcessorConfiguration(...)
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
        processorConfig: Qwen3VLProcessorConfiguration
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

    /// Extract and mean-pool patch embeddings from a single image into a 1D vector
    /// 
    /// This function preprocesses the image, applies patch embedding, and then mean-pools
    /// the resulting patch embeddings into a single 1D feature vector.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen3VL(config)
    /// let processorConfig = Qwen3VLProcessorConfiguration(...)
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
        processorConfig: Qwen3VLProcessorConfiguration
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

    /// Extract and mean-pool embeddings from a video by processing each frame
    /// 
    /// This function processes each frame of a video, extracts patch embeddings,
    /// mean-pools them, and returns an array of 1D feature vectors for each frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen3VL(config)
    /// let processorConfig = Qwen3VLProcessorConfiguration(...)
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
        processorConfig: Qwen3VLProcessorConfiguration
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

    /// Calculate cosine distance between two mean-pooled embeddings
    /// 
    /// This function computes the cosine distance between two 1D feature vectors
    /// obtained from mean-pooled patch embeddings. Cosine distance is 1 - cosine_similarity.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen3VL(config)
    /// let processorConfig = Qwen3VLProcessorConfiguration(...)
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

    /// Calculate cosine distance between each frame and the first frame as reference
    /// 
    /// This function extracts mean-pooled embeddings from each frame of a video
    /// and calculates cosine distance between each frame and the first frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen3VL(config)
    /// let processorConfig = Qwen3VLProcessorConfiguration(...)
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
        processorConfig: Qwen3VLProcessorConfiguration
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

    /// Extract and mean-pool embeddings from specific frames of a video
    /// 
    /// This function processes only the specified frames of a video, extracts patch embeddings,
    /// mean-pools them, and returns an array of 1D feature vectors for each specified frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen3VL(config)
    /// let processorConfig = Qwen3VLProcessorConfiguration(...)
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
        processorConfig: Qwen3VLProcessorConfiguration
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
    /// let model = Qwen3VL(config)
    /// let processorConfig = Qwen3VLProcessorConfiguration(...)
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
        processorConfig: Qwen3VLProcessorConfiguration
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

    /// Detect scene changes in a video using cosine distance threshold
    /// 
    /// This function analyzes each frame of a video and detects scene changes
    /// by comparing each frame to a reference frame. When distance exceeds
    /// the threshold, it marks a scene change and updates the reference frame.
    /// 
    /// Example usage:
    /// ```swift
    /// let model = Qwen3VL(config)
    /// let processorConfig = Qwen3VLProcessorConfiguration(...)
    /// let videoURL = URL(fileURLWithPath: "video.mp4")
    /// let sceneChanges = try model.detectSceneChanges(from: videoURL, threshold: 0.1, processorConfig: processorConfig)
    /// // sceneChanges contains frame indices where scene changes occur
    /// ```
    /// 
    /// - Parameter videoURL: The URL of the video file
    /// - Parameter threshold: Cosine distance threshold for scene change detection (default: 0.1)
    /// - Parameter minSceneDuration: Minimum scene duration in seconds (default: 2.0)
    /// - Parameter maxSceneDuration: Maximum scene duration in seconds (default: 15.0)
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
        processorConfig: Qwen3VLProcessorConfiguration
    ) async throws -> [(frameIndex: Int, timestamp: TimeInterval)] {
        let startTime = Date()
        
        // Extract CIImage frames from video at configured FPS for scene detection
        let ciImages = try await MediaProcessing.asCIImageSequence(
            AVAsset(url: videoURL), 
            samplesPerSecond: Int(processorConfig.fps)
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
            
            // Calculate timestamp for this frame
            let timestamp = TimeInterval(index) / processorConfig.fps
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

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let typedCache = castCacheOptional(cache)

        let result = languageModel(
            inputs,
            cache: typedCache,
            inputEmbeddings: nil,
            mask: nil,
            positionIds: nil,
            visualMask: nil,
            deepstackEmbeds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        ).logits
        return result
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var adjusted: [String: MLXArray] = [:]
        adjusted.reserveCapacity(weights.count)

        for (key, value) in weights {
            var newKey = key

            if newKey.contains("model") {
                if newKey.contains("model.visual") {
                    newKey = newKey.replacingOccurrences(of: "model.visual", with: "vision_tower")
                } else if newKey.contains("model.language_model") {
                    newKey = newKey.replacingOccurrences(
                        of: "model.language_model", with: "language_model.model")
                }
            } else if newKey.contains("lm_head") {
                newKey = newKey.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            if config.textConfiguration.tieWordEmbeddings && newKey.contains(".lm_head.") {
                // if using tieWordEmbeddings omit these keys as they will not be consumed
                continue
            }

            adjusted[newKey] = value
        }

        let sanitized = visionModel.sanitize(weights: adjusted)
        return sanitized
    }
}

extension Array where Element == THW {
    fileprivate var nilIfEmpty: [THW]? { isEmpty ? nil : self }
}

extension Qwen3VL {
    fileprivate func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    fileprivate func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }
}

public struct Qwen3VLMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        let imageContent = message.images.map { _ in
            ["type": "image"]
        }
        let textContent = [["type": "text", "text": message.content]]
        let videoContent = message.videos.map { _ in
            ["type": "video"]
        }

        return [
            "role": message.role.rawValue,
            "content": imageContent + videoContent + textContent,
        ]
    }
}
