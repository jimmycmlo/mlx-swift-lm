//
//  SmolVLM2.swift
//  mlx-swift-lm
//
//  Created by Pedro Cuenca on 20/3/25.
//

import AVFoundation
import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import Tokenizers

// MARK: - SmolVLM2 with FrameSpecification Support
//
// SmolVLM2 now supports selective video frame processing through the FrameSpecification enum.
// This allows you to process only specific frames from a video, which can be useful for:
// - Reducing processing time for long videos
// - Focusing on key moments or scenes
// - Analyzing specific timestamps or frame numbers
//
// Example usage:
// ```swift
// let processor = SmolVLMProcessor(config, tokenizer: tokenizer)
// 
// // Process all frames (default behavior)
// let allFramesInput = try await processor.prepare(input: userInput)
// 
// // Process specific frame numbers (0-based indexing)
// let specificFramesInput = try await processor.prepare(
//     input: userInput, 
//     frameSpecification: .frameNumbers([0, 10, 20, 30])
// )
// 
// // Process frames at specific timestamps (in seconds)
// let timestampFramesInput = try await processor.prepare(
//     input: userInput, 
//     frameSpecification: .timestamps([0.0, 5.0, 10.0, 15.0])
// )
// 
// // Process all frames explicitly
// let allFramesInput = try await processor.prepare(
//     input: userInput, 
//     frameSpecification: .allFrames
// )
// ```
//
// FrameSpecification options:
// - `.allFrames`: Process all frames in the video (default behavior)
// - `.frameNumbers([Int])`: Process specific frame numbers (0-based indexing)
// - `.timestamps([TimeInterval])`: Process frames at specific timestamps (in seconds)
//
// Notes:
// - Frame numbers are validated against video duration and FPS
// - Timestamps are validated against video duration
// - Invalid frame numbers or timestamps are automatically filtered out
// - The processor will log which frames are being processed for debugging

// MARK: - Configuration and modeling are Idefics3

typealias SmolVLM2Configuration = Idefics3Configuration
typealias SmolVLM2 = Idefics3

// MARK: - Frame Specification

/// Frame specification for selective video processing
public enum FrameSpecification {
    /// Process specific frame numbers (0-based indexing)
    case frameNumbers([Int])
    /// Process frames at specific timestamps (in seconds)
    case timestamps([TimeInterval])
    /// Process all frames (default behavior)
    case allFrames
}

// MARK: - SmolVLMProcessor and configuration

public struct SmolVLMProcessorConfiguration: Codable, Sendable {
    public struct Size: Codable, Sendable {
        public let longestEdge: Int
        enum CodingKeys: String, CodingKey {
            case longestEdge = "longest_edge"
        }
    }

    public struct VideoSampling: Codable, Sendable {
        public let fps: Int
        public let maxFrames: Int
        // Intentionally ignoring videoSize because I believe it's still wrong in the config files
        //        public let videoSize: Size

        enum CodingKeys: String, CodingKey {
            case fps
            case maxFrames = "max_frames"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size
    public let maxImageSize: Size
    public let videoSampling: VideoSampling
    private let _imageSequenceLength: Int?
    
    // Additional configuration properties for video processing
    private var _maxFrames: Int?
    private var _fps: Double?
    
    // TODO: this does not come in preprocessor_config.json, verify where transformers gets it from
    public var imageSequenceLength: Int { _imageSequenceLength ?? 64 }
    
    public var maxFrames: Int {
        get {
            _maxFrames ?? videoSampling.maxFrames
        }
        set {
            _maxFrames = newValue
        }
    }
    
    public var fps: Double {
        get {
            _fps ?? Double(videoSampling.fps)
        }
        set {
            _fps = newValue
        }
    }

    init(
        imageMean: [CGFloat], imageStd: [CGFloat], size: Size, maxImageSize: Size,
        videoSampling: VideoSampling, imageSequenceLength: Int?, maxFrames: Int? = nil, fps: Double? = nil
    ) {
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.size = size
        self.maxImageSize = maxImageSize
        self.videoSampling = videoSampling
        self._imageSequenceLength = imageSequenceLength
        self._maxFrames = maxFrames
        self._fps = fps
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
        case size
        case maxImageSize = "max_image_size"
        case videoSampling = "video_sampling"
        case _imageSequenceLength = "image_seq_len"
        case _maxFrames = "max_frames"
        case _fps = "fps"
    }
}

public class SmolVLMProcessor: UserInputProcessor {
    public var config: SmolVLMProcessorConfiguration
    private let tokenizer: any Tokenizer

    // FIXME: hardcoded values for now

    // Hardcode this since we can't pass it in or rely on it from the preprocessor config.
    let imageTokenId = 49190
    let imageToken = "<image>"
    let fakeImageToken = "<fake_token_around_image>"
    let globalImageToken = "<global-img>"

    var maxProcessingImageSize: CGFloat { CGFloat(config.size.longestEdge) }  // 2048
    var fixedImageSize: CGFloat { CGFloat(config.maxImageSize.longestEdge) }  // 384 for big models, 512 for small models (200-500M)
    var imageSequenceLength: Int { config.imageSequenceLength }
    var maxVideoFrames: Int { config.maxFrames }
    var targetVideoFPS: Double { config.fps }
    
    let defaultVideoSystemMessage =
        "You are a helpful assistant that can understand videos. Describe what type of video this is and what's happening in it."

    public init(
        _ config: SmolVLMProcessorConfiguration,
        tokenizer: any Tokenizer
    ) {
        self.config = config
        self.tokenizer = tokenizer
        print("SmolVLM2 Configuration:")
        print("  maxProcessingImageSize: \(maxProcessingImageSize)")
        print("  fixedImageSize: \(fixedImageSize)")
        print("  imageSequenceLength: \(imageSequenceLength)")
        print("  maxVideoFrames: \(maxVideoFrames)")
        print("  targetVideoFPS: \(targetVideoFPS)")
        print("  config.maxFrames: \(config.maxFrames)")
        print("  config.fps: \(config.fps)")
    }

    func getVideoPromptString(
        frameCount: Int, timeStamps: [String], videoDuration: String, seqLen: Int,
        fakeToken: String, imageToken: String, globalImageToken: String
    ) -> String {
        var textSplitFrames =
            "You are provided the following series of \(frameCount) frames from a \(videoDuration) [H:MM:SS] video.\n"
        for frameIndex in 0 ..< frameCount {
            textSplitFrames += "\nFrame from \(timeStamps[frameIndex]):"
            textSplitFrames +=
                (fakeToken
                    + globalImageToken
                    + String(repeating: imageToken, count: seqLen)
                    + fakeToken)
        }
        textSplitFrames += "\n\n"
        return textSplitFrames
    }

    func getImagePromptString(
        rows: Int, cols: Int, seqLen: Int, fakeToken: String, imageToken: String,
        globalImageToken: String
    ) -> String {
        /// Prompt with expanded image tokens for when the image is split into patches.
        /// This applies to image processing, not video (I think).
        /// This just transliterates this: https://github.com/huggingface/transformers/blob/6a1ab634b6886b6560b0502e7a305c8cd881732e/src/transformers/models/idefics3/processing_idefics3.py#L44
        var textSplitImages = ""
        for h in 0 ..< rows {
            for w in 0 ..< cols {
                textSplitImages +=
                    (fakeToken
                        + "<row_\(h + 1)_col_\(w + 1)>"
                        + String(repeating: imageToken, count: seqLen))
            }
            textSplitImages += "\n"
        }
        textSplitImages +=
            ("\n"
                + fakeToken
                + globalImageToken
                + String(repeating: imageToken, count: seqLen)
                + fakeToken)
        return textSplitImages
    }

    /// Compute the resize size with `longestEdge` for the given size
    /// If `multiple` is not nil, ensures each side is a multiple of that value
    func aspectRatioSize(for size: CGSize, longestEdge: CGFloat, multiple: CGFloat? = nil) -> CGSize
    {
        var targetSize = MediaProcessing.bestFit(
            size, in: CGSize(width: longestEdge, height: longestEdge))
        guard let multiple = multiple else { return targetSize }
        let aspectRatio = targetSize.width / targetSize.height
        if size.width >= size.height {
            let width = ceil(targetSize.width / multiple) * multiple
            var height = width / aspectRatio
            height = ceil(height / multiple) * multiple
            return CGSize(width: width, height: height)
        } else {
            let height = ceil(targetSize.height / multiple) * multiple
            var width = height * aspectRatio
            width = ceil(width / multiple) * multiple
            return CGSize(width: width, height: height)
        }
    }

    /// Compute the resize size with `longestEdge` for the given size
    /// If `multiple` is not nil, ensures each side is a multiple of that value
    func aspectRatioSize(for size: CGSize, longestEdge: Int, multiple: Int? = nil) -> CGSize {
        return aspectRatioSize(
            for: size, longestEdge: CGFloat(longestEdge), multiple: multiple.flatMap(CGFloat.init))
    }

    /// Tile image if it's larger than the maxProcessingImageSize, so the model gets to see more of it
    /// TODO: disable in video mode
    func tiles(from originalImage: CIImage) -> (tiles: [CIImage], rows: Int, cols: Int) {
        // The original code resizes to maxProcessingImageSize, then resizes again ensuring multiples of fixedImageSize
        // We do both resizes in one go
        let processingSize = aspectRatioSize(
            for: originalImage.extent.size, longestEdge: maxProcessingImageSize,
            multiple: fixedImageSize)
        let image = MediaProcessing.resampleLanczos(originalImage, to: processingSize)

        var tiles: [CIImage] = []

        // Crop nRows x nCols tiles
        let nRows = Int(ceil(image.extent.size.height / CGFloat(fixedImageSize)))
        let nCols = Int(ceil(image.extent.size.width / CGFloat(fixedImageSize)))

        // Warning: in CIImage, y=0 is the bottom side. We reverse the rows to match the transformers processor
        let tileEdge = Int(fixedImageSize)
        for row in (0 ..< nRows).reversed() {
            for col in 0 ..< nCols {
                let x0 = col * tileEdge
                let y0 = row * tileEdge
                let x1 = min(x0 + tileEdge, Int(image.extent.size.width))
                let y1 = min(y0 + tileEdge, Int(image.extent.size.height))

                let tile = image.cropped(to: CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0))
                tiles.append(tile)
            }
        }

        return (tiles, nRows, nCols)
    }

    func formatTimestamp(_ time: CMTime) -> String {
        let totalSeconds = Int(ceil(time.seconds))
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let seconds = totalSeconds % 60

        return String(format: "%d:%02d:%02d", hours, minutes, seconds)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Default to processing all frames
        return try await prepare(input: input, frameSpecification: .allFrames)
    }
    
    /// Prepare input with specific frame specification for selective video processing
    /// - Parameters:
    ///   - input: The user input containing text, images, and/or videos
    ///   - frameSpecification: The frame specification for video processing
    /// - Returns: Prepared LMInput ready for model inference
    public func prepareWithFrameSpecification(input: UserInput, frameSpecification: FrameSpecification) async throws -> LMInput {
        return try await prepare(input: input, frameSpecification: frameSpecification)
    }
    
    public func prepare(input: UserInput, frameSpecification: FrameSpecification) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)  // TODO: Create SmolVLM2MessageGenerator

        if input.images.isEmpty && input.videos.isEmpty {
            // No image scenario
            let promptTokens = try tokenizer.applyChatTemplate(messages: messages)
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: tokensArray)
            return LMInput(text: .init(tokens: tokensArray, mask: mask), image: nil)
        } else if input.images.count > 0 && input.videos.isEmpty {
            // Single image scenario
            guard input.images.count == 1 else {
                throw VLMError.singleImageAllowed
            }

            // Unfortunately we don't have a "render" option in Tokenizers yet, so decoding
            let promptTokens = try tokenizer.applyChatTemplate(messages: messages)
            let decoded = try tokenizer.decode(tokens: promptTokens, skipSpecialTokens: false)

            let image = try input.images[0].asCIImage().toSRGB()
            let (tiles, imageRows, imageCols) = tiles(from: image)

            // Append the resized global image
            // Note we are resampling from the original (potentially larger), not the processing size. It shouldn't make much difference.
            let images =
                tiles + [
                    image.resampled(
                        to: CGSize(width: fixedImageSize, height: fixedImageSize), method: .lanczos)
                ]

            let pixelsForImages = images.map {
                $0.normalized(mean: config.imageMeanTuple, std: config.imageStdTuple).asMLXArray()
            }

            // In transformers we have a batch dim plus the number of images per batch, and they get collapsed inside the model.
            // Here we provide the compact version.
            let pixels = concatenated(pixelsForImages, axis: 0).transposed(0, 2, 3, 1)

            let imagePromptString = getImagePromptString(
                rows: imageRows,
                cols: imageCols,
                seqLen: imageSequenceLength,
                fakeToken: fakeImageToken,
                imageToken: imageToken,
                globalImageToken: globalImageToken
            )

            let prompt = decoded.replacingOccurrences(of: imageToken, with: imagePromptString)
            let finalPromptTokens = try tokenizer.encode(text: prompt)

            let promptArray = MLXArray(finalPromptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray)

            return LMInput(
                text: .init(tokens: promptArray, mask: mask),
                image: .init(pixels: pixels)
            )
        } else {
            // Single video scenario
            guard input.images.count == 0 else {
                throw VLMError.singleMediaTypeAllowed
            }
            guard input.videos.count == 1 else {
                throw VLMError.singleVideoAllowed
            }

            // Insert a default system message if the input doesn't have one
            func messagesWithSystem(_ messages: [Message]) -> [Message] {
                guard messages.filter { $0["role"] as? String == "system" }.isEmpty else {
                    return messages
                }

                var messagesWithSystem = messages
                messagesWithSystem.insert(
                    [
                        "role": "system",
                        "content": [["type": "text", "text": defaultVideoSystemMessage]],
                    ], at: 0)
                return messagesWithSystem
            }

            // Unfortunately we don't have a "render" option in Tokenizers yet, so decoding
            let finalMessages = messagesWithSystem(messages)
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messagesWithSystem(messages))
            let decoded = try tokenizer.decode(tokens: promptTokens, skipSpecialTokens: false)

            var video = try input.videos[0].asAVAsset()

            // Handle frame specification for selective video processing
            let processedFrames: ProcessedFrames
            switch frameSpecification {
            case .allFrames:
                print("SmolVLM2: Using ALL FRAMES processing mode")
                // Process all frames as before
                var frameCounter = 0
                processedFrames = await try MediaProcessing.asProcessedSequence(
                    video,
                    maxFrames: maxVideoFrames,
                    targetFPS: { duration in
                        // 1 fps for duration >= 10s, apply a multiplier if smaller
                        max((10 - 0.9 * duration.seconds) * targetVideoFPS, 1)
                    }
                ) { frame in
                    // Log frame information
                    let timestamp = frame.timeStamp
                    let size = frame.frame.extent.size
//                    print("SmolVLM2: Processing frame \(frameCounter) at \(formatTimestamp(timestamp)) (size: \(Int(size.width))x\(Int(size.height)))")
                    frameCounter += 1
                    
                    let processedFrame = frame.frame
                        .toSRGB()
                        .resampled(
                            to: CGSize(width: fixedImageSize, height: fixedImageSize), method: .lanczos
                        )
                        .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
                    print("SmolVLM2: Processing frame \(frameCounter) at \(formatTimestamp(timestamp)) (size: \(Int(fixedImageSize))x\(Int(fixedImageSize)))")
                    return VideoFrame(frame: processedFrame, timeStamp: frame.timeStamp)
                }
                
            case .frameNumbers(let frameNumbers):
                print("SmolVLM2: Using FRAME NUMBERS processing mode")
                // Process specific frame numbers
                let sortedFrameNumbers = frameNumbers.sorted()
                let duration = try await video.load(.duration)
                let durationSeconds = CMTimeGetSeconds(duration)
                let maxFrameNumber = Int(durationSeconds * targetVideoFPS)
                
                let validFrames = sortedFrameNumbers.filter { $0 >= 0 && $0 < maxFrameNumber }
                print("SmolVLM2: Processing specific frame numbers: \(validFrames)")
                
                var selectedFrames: [MLXArray] = []
                var selectedTimestamps: [CMTime] = []
                
                for frameNumber in validFrames {
                    let timestamp = TimeInterval(frameNumber) / targetVideoFPS
                    let time = CMTime(seconds: timestamp, preferredTimescale: 600)
                    
                    // Extract frame using AVAssetImageGenerator
                    let generator = AVAssetImageGenerator(asset: video)
                    generator.appliesPreferredTrackTransform = true
                    generator.requestedTimeToleranceBefore = .zero
                    generator.requestedTimeToleranceAfter = .zero
                    
                    let cgImage = try await generator.image(at: time).image
                    let frameImage = CIImage(cgImage: cgImage, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
                    
                    // Log frame information
                    let size = frameImage.extent.size
                    print("SmolVLM2: Processing frame \(frameNumber) at \(formatTimestamp(time)) (size: \(Int(size.width))x\(Int(size.height)))")
                    
                    let processedFrame = frameImage
                        .toSRGB()
                        .resampled(
                            to: CGSize(width: fixedImageSize, height: fixedImageSize), method: .lanczos
                        )
                        .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
                    
                    selectedFrames.append(processedFrame.asMLXArray())
                    selectedTimestamps.append(time)
                }
                
                processedFrames = ProcessedFrames(
                    frames: selectedFrames,
                    timestamps: selectedTimestamps,
                    totalDuration: duration
                )
                
            case .timestamps(let timestamps):
                print("SmolVLM2: Using TIMESTAMPS processing mode")
                // Process frames at specific timestamps
                let sortedTimestamps = timestamps.sorted()
                let duration = try await video.load(.duration)
                let durationSeconds = CMTimeGetSeconds(duration)
                
                let validTimestamps = sortedTimestamps.filter { $0 >= 0 && $0 <= durationSeconds }
                print("SmolVLM2: Processing frames at timestamps: \(validTimestamps.map { String(format: "%.2f", $0) })")
                
                var selectedFrames: [MLXArray] = []
                var selectedTimestamps: [CMTime] = []
                
                for (index, timestamp) in validTimestamps.enumerated() {
                    let time = CMTime(seconds: timestamp, preferredTimescale: 600)
                    
                    // Extract frame using AVAssetImageGenerator
                    let generator = AVAssetImageGenerator(asset: video)
                    generator.appliesPreferredTrackTransform = true
                    generator.requestedTimeToleranceBefore = .zero
                    generator.requestedTimeToleranceAfter = .zero
                    
                    let cgImage = try await generator.image(at: time).image
                    let frameImage = CIImage(cgImage: cgImage, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
                    
                    // Log frame information
                    let size = frameImage.extent.size
                    print("SmolVLM2: Processing frame \(index) at \(formatTimestamp(time)) (size: \(Int(size.width))x\(Int(size.height)))")
                    
                    let processedFrame = frameImage
                        .toSRGB()
                        .resampled(
                            to: CGSize(width: fixedImageSize, height: fixedImageSize), method: .lanczos
                        )
                        .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
                    
                    selectedFrames.append(processedFrame.asMLXArray())
                    selectedTimestamps.append(time)
                }
                
                processedFrames = ProcessedFrames(
                    frames: selectedFrames,
                    timestamps: selectedTimestamps,
                    totalDuration: duration
                )
            }

            let thwFrames = (0 ..< processedFrames.frames.count).map {
                THW($0, Int(fixedImageSize), Int(fixedImageSize))
            }

            let stackedFrames = concatenated(processedFrames.frames, axis: 0)
            let transposedFrames = stackedFrames.transposed(0, 2, 3, 1)

            let videoPromptString = getVideoPromptString(
                frameCount: processedFrames.frames.count,
                timeStamps: processedFrames.timestamps.map(formatTimestamp),
                videoDuration: formatTimestamp(processedFrames.totalDuration),
                seqLen: imageSequenceLength,
                fakeToken: fakeImageToken, imageToken: imageToken,
                globalImageToken: globalImageToken)

            let prompt: String
            if let range = decoded.range(of: "User: ") {
                let before = decoded[..<range.upperBound]
                let after = decoded[range.upperBound...]
                prompt = String(before) + videoPromptString + String(after)
            } else {
                // Fallback if the expected marker is not present
                prompt = decoded + "\n" + videoPromptString
            }
            let finalPromptTokens = try tokenizer.encode(text: prompt)

            let promptArray = MLXArray(finalPromptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray)
            return LMInput(
                text: .init(tokens: promptArray, mask: mask),
                image: .init(pixels: transposedFrames, frames: thwFrames)
            )
        }
    }
}
