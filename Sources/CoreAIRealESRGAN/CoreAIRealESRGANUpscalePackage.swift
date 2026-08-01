// CoreAIRealESRGANUpscalePackage.swift
//
// The MLXEngine `imageUpscale` package over the CoreAI/ANE core — the SECOND backend behind the
// capability the MLX sibling already serves, registerable beside it under its own PackageID.
//
// 🔑 **This product carries no MLX.** It depends on MLXToolKit alone — the engine's dependency-free
// contract layer — so "an engine package" and "an MLX package" are finally distinct things, which is
// exactly what the engine's runtime-agnostic capability model promised (`audioPolish` proved the
// weightless seam; this proves the ANE one).
//
// ⚠️ macOS 27+ (CoreAI.framework; public release expected October per Apple's cadence). ForgeCore
// floors at 26 and therefore does NOT link this — a host app injects it through ForgeCore's
// external-registration seam when its own deployment target allows. The MLX sibling remains the
// fallback below 27; that OS split is the design, not a limitation.
//
// Scale semantics, response formats, and sub-native downsampling mirror the MLX sibling exactly —
// the two backends must be interchangeable behind a planner, so any behavioural divergence here is
// a defect, not a feature.

import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import ImageIO
import MLXToolKit
import RealESRGANCoreAI
import UniformTypeIdentifiers

public enum CoreAIRealESRGANPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
}

/// Configuration: variant + compute preference. Codable so hosts can persist it; both fields are
/// portable (unlike a byte budget, a variant IS a property of the configuration).
public struct CoreAIRealESRGANConfiguration: PackageConfiguration, Sendable, Codable {
    public var variant: Variant
    public var compute: Compute

    public enum Variant: String, Sendable, Codable, CaseIterable {
        case general, generalDenoise, anime

        var coreVariant: SRVGGNetCompact_CoreAI.Variant {
            switch self {
            case .general:        return .general
            case .generalDenoise: return .generalWDN
            case .anime:          return .anime
            }
        }
    }

    public enum Compute: String, Sendable, Codable {
        case neuralEngine, gpu

        var coreCompute: SRVGGNetCompact_CoreAI.Compute {
            self == .neuralEngine ? .neuralEngine : .gpu
        }
    }

    public init(variant: Variant = .general, compute: Compute = .neuralEngine) {
        self.variant = variant
        self.compute = compute
    }
}

@InferenceActor
public final class CoreAIRealESRGANUpscalePackage: ModelPackage {
    public typealias Configuration = CoreAIRealESRGANConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Same licensing as the MLX sibling: xinntao weights BSD-3, port code MIT.
            license: LicenseDeclaration(weightLicense: .bsd3, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "coreai-community/Real-ESRGAN-CoreAI",
                                   revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // MEASURED via `phys_footprint` (the admission basis) through this package surface,
                // 1920×1080 → ×4 (7680×4320 out), release: baseline 3 MB → after load 19 MB
                // (model + E5RT executable) → peak 0.86 GB, 1.60 s incl. the 33 MP PNG encode.
                // The working set is the float32 accumulation planes (16 B per OUTPUT pixel:
                // 3 colour + 1 weight), so activation scales with OUTPUT area — there is no
                // whole-frame intermediate; the ANE keeps layer activations on-die. That makes this
                // backend ~25× lighter than the MLX sibling's whole-frame path at 1080p (21.24 GB)
                // and ~7× lighter than its tiled path (~6 GB). Declared with headroom over the
                // measurement; fp16 is the only quant an .aimodel ships.
                footprints: [QuantFootprint(quant: .fp16,
                                            residentBytes: 25_000_000,
                                            peakActivationBytes: 900_000_000)],
                // ⚠️ `.coreMLANE` is the engine's ANE placement value, named before CoreAI existed.
                // It means "the Neural Engine pool"; the framework driving it is this package's
                // concern, not the governor's. An engine-side rename (`aneCoreAI`?) is queued.
                requiredBackends: [.coreMLANE],
                os: OSRequirement(minMacOS: SemanticVersion(major: 27, minor: 0, patch: 0)),
                chipFloor: nil   // every Apple-silicon Mac has an ANE
            ),
            specialties: [],
            surfaces: [
                ImageUpscaleContract.descriptor(
                    name: "coreai-realesrgan-upscale",
                    summary: "Real-ESRGAN 4x super-resolution on the Apple Neural Engine (CoreAI), tile-based."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var upscaler: SRVGGNetCompact_CoreAI?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard upscaler == nil else { return }
        let model = SRVGGNetCompact_CoreAI(variant: configuration.variant.coreVariant,
                                           compute: configuration.compute.coreCompute)
        // Pay the E5RT specialization at load (MAT-gate semantics: preparation is visible,
        // first inference is not secretly slow). ~8 s cold per machine, OS-cached after.
        try await model.prepare()
        upscaler = model
    }

    public func unload() async {
        // No MLX pool to flush — the executable and its ANE working set are owned by the OS's
        // E5RT cache, and dropping the reference releases the process-side buffers.
        upscaler = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint first; mid-run cadence is per tile inside the core.
        try Task.checkCancellation()
        guard let upscaler else { throw PackageError.notLoaded }
        guard request.capability == .imageUpscale,
              let req = request as? ImageUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let input = try Self.decodeToCGImage(req.image)
        let native = SRVGGNetCompact_CoreAI.scaleFactor

        let nativeOut = try await upscaler.upscale(input)

        // Sub-native scale honored by post-downsampling, `appliedScale` reports what ran —
        // byte-for-byte the sibling's contract.
        let out: CGImage
        let appliedScale: Int
        if let s = req.scale, s > 0, s < native {
            out = try Self.resize(nativeOut, width: input.width * s, height: input.height * s)
            appliedScale = s
        } else {
            out = nativeOut
            appliedScale = native
        }

        guard let png = Self.encodePNG(out) else { throw CoreAIRealESRGANPackageError.imageEncodeFailed }
        return ImageUpscaleResponse(
            image: Image(format: .png, data: png, width: out.width, height: out.height),
            appliedScale: appliedScale)
    }

    // MARK: - Codec

    nonisolated static func decodeToCGImage(_ image: Image) throws -> CGImage {
        if image.format == .rawBGRA8 {
            guard let w = image.width, let h = image.height else {
                throw CoreAIRealESRGANPackageError.imageDecodeFailed("rawBGRA8 requires width/height")
            }
            let stride = image.bytesPerRow ?? (w * 4)
            guard let provider = CGDataProvider(data: image.data as CFData),
                  let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                                   bytesPerRow: stride, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue
                                       | CGBitmapInfo.byteOrder32Little.rawValue),
                                   provider: provider, decode: nil, shouldInterpolate: false,
                                   intent: .defaultIntent) else {
                throw CoreAIRealESRGANPackageError.imageDecodeFailed("rawBGRA8 wrap failed")
            }
            return cg
        }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw CoreAIRealESRGANPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        return cg
    }

    nonisolated static func encodePNG(_ cg: CGImage) -> Data? {
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    nonisolated static func resize(_ cg: CGImage, width: Int, height: Int) throws -> CGImage {
        guard width > 0, height > 0,
              let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw CoreAIRealESRGANPackageError.imageEncodeFailed
        }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let out = ctx.makeImage() else { throw CoreAIRealESRGANPackageError.imageEncodeFailed }
        return out
    }
}

extension CoreAIRealESRGANUpscalePackage {
    public nonisolated static var registration: PackageRegistration {
        .of(CoreAIRealESRGANUpscalePackage.self)
    }
}
