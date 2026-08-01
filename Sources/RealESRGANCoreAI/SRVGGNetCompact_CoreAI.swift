// SRVGGNetCompact_CoreAI.swift
//
// Role: Real-ESRGAN 4× super-resolution on the Apple Neural Engine, tiled at the fixed 128/8
//       geometry the vendored `.aimodel` assets were exported at.
//
// Upstream: https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause). Weights are the ORIGINAL
// released checkpoints, re-exported PyTorch → `coreai-torch` → `.aimodel` (scripts/ in this repo —
// the export is reproducible, not a binary of unknown provenance). ⚠️ This is a RE-PORT from the
// original weights, not a conversion of the MLX port; only the parity methodology transfers.
//
// Parity, measured per variant before anything shipped (fp16 on ANE vs fp32 torch, 7 real
// 128² tiles — 5 signage stills + 2 video-master frames; the sibling-package lesson is that dtype
// verdicts can INVERT between variants, so each is measured, none assumed):
//
//     general      min 68.56 / mean 69.36 dB
//     generalWDN   min 58.15 / mean 65.09 dB   (one photo outlier; still 8 dB above the
//                                               FFTformer-fp16 precedent bar of ~50 dB)
//     anime        min 64.11 / mean 69.43 dB
//
// 🔑 Compositing is the NORMALIZED FEATHERED blend (Σwᵢxᵢ/Σwᵢ in float32, quantize once), not the
// probe's center-write — the V10-fix/V11 lesson is that the compositor is where tiled paths break,
// and this one follows the shipping MLX compositor's contract exactly.
//
// ⚠️ First use per machine pays one E5RT specialization (~8 s, OS-cached thereafter). Callers
// should treat the first `upscale` as a warm-up, never profile it (the standing rule), and surface
// it as model preparation in a UI.

import CoreAI
import CoreGraphics
import Foundation

/// SRVGGNetCompact on the ANE — fp16, static 128² tiles, 4× fixed.
public final class SRVGGNetCompact_CoreAI: @unchecked Sendable {

    /// Vendored variant. Same three checkpoints as the MLX sibling, same roles.
    public enum Variant: String, Sendable, CaseIterable {
        case general        // realesr-general-x4v3   (num_conv 32)
        case generalWDN     // realesr-general-wdn-x4v3 — denoising variant
        case anime          // realesr-animevideov3   (num_conv 16)

        var assetName: String {
            switch self {
            case .general:    return "realesr_general_x4v3_float16_static128"
            case .generalWDN: return "realesr_general_wdn_x4v3_float16_static128"
            case .anime:      return "realesr_animevideov3_float16_static128"
            }
        }
    }

    /// Where to run. `.neuralEngine` is the point of this package (the measured ≈4.5–4.9× energy
    /// win); `.gpu` exists because CoreAI's static executables also measured 2.2–2.4× faster than
    /// MLX dynamic GPU at parity (V13 ②) — a legitimate second mode, not a debug switch.
    public enum Compute: String, Sendable {
        case neuralEngine, gpu, cpu

        var options: SpecializationOptions {
            switch self {
            case .neuralEngine: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
            case .gpu:          return SpecializationOptions(preferredComputeUnitKind: .gpu)
            case .cpu:          return SpecializationOptions(preferredComputeUnitKind: .cpu)
            }
        }
    }

    /// Fixed by the static export. Not injectable on purpose — the ANE compiles one executable per
    /// shape, so geometry is a build-time property of the asset, the inversion of the MLX sibling.
    public static let tileSize = 128
    public static let tileOverlap = 8
    public static let scaleFactor = 4

    public let variant: Variant
    public let compute: Compute

    // Loaded lazily; `withLock` closures keep NSLock legal from async contexts (never held across
    // an await — the load itself happens outside the lock and only the pointer swap is guarded).
    private var function: InferenceFunction?
    private let loadLock = NSLock()

    public init(variant: Variant, compute: Compute = .neuralEngine) {
        self.variant = variant
        self.compute = compute
    }

    // MARK: - Loading

    public enum CoreAIUpscaleError: Error, CustomStringConvertible {
        case assetMissing(String)
        case functionMissing
        case badImage
        public var description: String {
            switch self {
            case .assetMissing(let n): return "bundled .aimodel missing: \(n) — broken build product"
            case .functionMissing:     return "no 'main' function in the .aimodel"
            case .badImage:            return "could not read input pixels"
            }
        }
    }

    /// Load + specialize. Safe to call repeatedly; the first call per machine is the slow one
    /// (E5RT specialization, OS-cached).
    public func prepare() async throws {
        if loadLock.withLock({ function != nil }) { return }

        guard let url = Bundle.module.url(forResource: variant.assetName, withExtension: "aimodel")
        else { throw CoreAIUpscaleError.assetMissing(variant.assetName) }
        let model = try await AIModel(contentsOf: url, options: compute.options)
        guard let fn = try model.loadFunction(named: "main") else {
            throw CoreAIUpscaleError.functionMissing
        }
        loadLock.withLock { function = fn }
    }

    // MARK: - Inference

    /// Upscale 4×. Any input size — tiled at 128/8 with clamped-and-deduplicated origins and a
    /// normalized feathered composite.
    public func upscale(_ image: CGImage) async throws -> CGImage {
        try await prepare()
        guard let fn = loadLock.withLock({ function }) else { throw CoreAIUpscaleError.functionMissing }

        let W = image.width, H = image.height
        guard let chw = Self.planarCHW(from: image) else { throw CoreAIUpscaleError.badImage }

        let tile = Self.tileSize, scale = Self.scaleFactor
        let xs = Self.origins(extent: W, tile: tile, overlap: Self.tileOverlap)
        let ys = Self.origins(extent: H, tile: tile, overlap: Self.tileOverlap)

        let oW = W * scale, oH = H * scale
        var colour = [Float](repeating: 0, count: 3 * oH * oW)
        var weight = [Float](repeating: 0, count: oH * oW)
        let ramp = Self.rampWeights(tile: tile * scale, overlap: Self.tileOverlap * scale)

        var input = NDArray(shape: [1, 3, tile, tile], scalarType: .float16)

        for ty in ys {
            for tx in xs {
                // Fill the tile (edge-clamped reads for the partial tiles at the frame border).
                input.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, _ in
                    for c in 0 ..< 3 {
                        for y in 0 ..< tile {
                            let sy = min(ty + y, H - 1)
                            for x in 0 ..< tile {
                                let sx = min(tx + x, W - 1)
                                ptr[(c * tile + y) * tile + x] = chw[c * H * W + sy * W + sx]
                            }
                        }
                    }
                }
                var outputs = try await fn.run(inputs: ["x": input])
                guard let outValue = outputs.remove("output") ?? outputs.remove("out"),
                      let nd = outValue.ndArray else { throw CoreAIUpscaleError.functionMissing }

                // Accumulate feathered (float32; quantize exactly once at readout).
                let ox = tx * scale, oy = ty * scale, ot = tile * scale
                nd.view(as: Float16.self).withUnsafePointer { src, _, _ in
                    for y in 0 ..< ot {
                        let gy = oy + y
                        guard gy < oH else { break }
                        for x in 0 ..< ot {
                            let gx = ox + x
                            guard gx < oW else { break }
                            let w = ramp[y] * ramp[x]
                            weight[gy * oW + gx] += w
                            for c in 0 ..< 3 {
                                colour[c * oH * oW + gy * oW + gx]
                                    += w * Float(src[(c * ot + y) * ot + x])
                            }
                        }
                    }
                }
            }
        }
        return Self.image(colour: colour, weight: weight, width: oW, height: oH)
    }

    // MARK: - Geometry (mirrors MLXTileProcessor: clamped stride grid, duplicates collapsed)

    static func origins(extent: Int, tile: Int, overlap: Int) -> [Int] {
        let step = max(tile - overlap, 1)
        var out: [Int] = []
        var p = 0
        while true {
            let clamped = max(0, min(p, extent - tile))
            if out.last != clamped { out.append(clamped) }
            if p + tile >= extent { break }
            p += step
        }
        return out
    }

    /// Symmetric edge ramp over the OVERLAP width — interior weight 1, linear feather at both ends.
    static func rampWeights(tile: Int, overlap: Int) -> [Float] {
        var w = [Float](repeating: 1, count: tile)
        guard overlap > 0 else { return w }
        for i in 0 ..< overlap {
            let v = Float(i + 1) / Float(overlap + 1)
            w[i] = v
            w[tile - 1 - i] = v
        }
        return w
    }

    // MARK: - Pixels

    static func planarCHW(from image: CGImage) -> [Float16]? {
        let W = image.width, H = image.height
        var rgba = [UInt8](repeating: 0, count: W * H * 4)
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: &rgba, width: W, height: H, bitsPerComponent: 8,
                                  bytesPerRow: W * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: W, height: H))
        var chw = [Float16](repeating: 0, count: 3 * H * W)
        for y in 0 ..< H {
            for x in 0 ..< W {
                let p = (y * W + x) * 4
                chw[0 * H * W + y * W + x] = Float16(Float(rgba[p]) / 255)
                chw[1 * H * W + y * W + x] = Float16(Float(rgba[p + 1]) / 255)
                chw[2 * H * W + y * W + x] = Float16(Float(rgba[p + 2]) / 255)
            }
        }
        return chw
    }

    static func image(colour: [Float], weight: [Float], width: Int, height: Int) -> CGImage {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let idx = y * width + x
                let w = max(weight[idx], .leastNormalMagnitude)
                for c in 0 ..< 3 {
                    let v = colour[c * height * width + idx] / w
                    rgba[idx * 4 + c] = UInt8(max(0, min(255, (v * 255).rounded())))
                }
            }
        }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(data: &rgba, width: width, height: height, bitsPerComponent: 8,
                            bytesPerRow: width * 4, space: cs,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return ctx.makeImage()!
    }
}
