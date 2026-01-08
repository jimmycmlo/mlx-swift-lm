// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Notification sent when a model download completes.
///
/// The `userInfo` dictionary contains:
/// - `"modelDirectory"`: The `URL` of the downloaded model directory
/// - `"configuration"`: The `ModelConfiguration` that was downloaded
public extension Notification.Name {
    static let modelDownloadCompleted = Notification.Name("modelDownloadCompleted")
}

/// Download the model using the `HubApi`.
///
/// This will download `*.safetensors` and `*.json` if the ``ModelConfiguration``
/// represents a Hub id, e.g. `mlx-community/gemma-2-2b-it-4bit`.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``
///
/// - Parameters:
///   - hub: HubApi instance
///   - configuration: the model identifier
///   - progressHandler: callback for progress
///   - completionHandler: optional callback called when download completes
/// - Returns: URL for the directory containing downloaded files
public func downloadModel(
    hub: HubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
    completionHandler: (@Sendable (URL) -> Void)? = nil
) async throws -> URL {
    do {
        let result: URL
        switch configuration.id {
        case .id(let id, let revision):
            // download the model weights
            let repo = Hub.Repo(id: id)
            let modelFiles = ["*.safetensors", "*.json"]
            
            // Wrap progress handler to detect completion
            var lastProgress: Double = 0.0
            let wrappedProgressHandler: @Sendable (Progress) -> Void = { progress in
                progressHandler(progress)
                // Check if download completed (progress reaches 1.0 or is completed)
                let currentProgress = progress.fractionCompleted
                if currentProgress >= 1.0 && lastProgress < 1.0 {
                    // Download just completed
                }
                lastProgress = currentProgress
            }
            
            result = try await hub.snapshot(
                from: repo,
                revision: revision,
                matching: modelFiles,
                progressHandler: wrappedProgressHandler
            )
            
            // Post notification that download completed
            NotificationCenter.default.post(
                name: .modelDownloadCompleted,
                object: nil,
                userInfo: [
                    "modelDirectory": result,
                    "configuration": configuration
                ]
            )
            
            // Call completion handler after successful download
            if let completionHandler {
                completionHandler(result)
            }
            
            return result
        case .directory(let directory):
            // For local directories, call completion immediately
            if let completionHandler {
                completionHandler(directory)
            }
            return directory
        }

    } catch Hub.HubClientError.authorizationRequired {
        // an authorizationRequired means (typically) that the named repo doesn't exist on
        // on the server so retry with local only configuration
        let directory = configuration.modelDirectory(hub: hub)
        if let completionHandler {
            completionHandler(directory)
        }
        return directory

    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
            // Error Domain=NSURLErrorDomain Code=-1009 "The Internet connection appears to be offline."
            // fall back to the local directory
            let directory = configuration.modelDirectory(hub: hub)
            if let completionHandler {
                completionHandler(directory)
            }
            return directory
        } else {
            throw error
        }
    }
}

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:)``, applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    // load the weights
    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }
    }

    // per-model cleanup
    weights = model.sanitize(weights: weights)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
}
