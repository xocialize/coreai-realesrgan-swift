// Geometry + compositor arithmetic run everywhere (pure CPU, weightless). Live ANE inference is
// gated on the framework actually loading — CI runners below macOS 27 skip it honestly rather than
// green-lighting a path they never exercised.
import CoreGraphics
import Foundation
import Testing

@testable import RealESRGANCoreAI

@Suite("Tile geometry")
struct GeometryTests {
    /// The clamped grid with duplicates collapsed — the exact contract MLXTileProcessor settled on
    /// after the duplicate-origin defect (its v0.6.4 lesson, inherited here rather than re-learned).
    @Test func originsClampAndDedupe() {
        // Exactly one tile when the extent fits.
        #expect(SRVGGNetCompact_CoreAI.origins(extent: 128, tile: 128, overlap: 8) == [0])
        // Smaller than a tile: single clamped origin at 0, never negative.
        #expect(SRVGGNetCompact_CoreAI.origins(extent: 96, tile: 128, overlap: 8) == [0])
        // 256 wide: 0, 120, then 240→clamped 128. No duplicates.
        let o = SRVGGNetCompact_CoreAI.origins(extent: 256, tile: 128, overlap: 8)
        #expect(o == [0, 120, 128])
        #expect(Set(o).count == o.count)
        // Every pixel covered.
        for e in [96, 128, 200, 256, 300, 511, 1920] {
            let os = SRVGGNetCompact_CoreAI.origins(extent: e, tile: 128, overlap: 8)
            var covered = [Bool](repeating: false, count: e)
            for p in os { for i in p ..< min(p + 128, e) { covered[i] = true } }
            #expect(!covered.contains(false), "gap in coverage at extent \(e)")
        }
    }

    /// Feather weights: symmetric, interior 1, strictly positive — Σw normalization then divides
    /// away any absolute scale, but a zero weight would divide by nothing at the frame corner.
    @Test func rampIsSymmetricAndPositive() {
        let w = SRVGGNetCompact_CoreAI.rampWeights(tile: 512, overlap: 32)
        #expect(w.count == 512)
        #expect(w[256] == 1)
        #expect(w.allSatisfy { $0 > 0 })
        for i in 0 ..< 32 { #expect(abs(w[i] - w[511 - i]) < 1e-6) }
    }
}

@Suite("Compositor")
struct CompositorTests {
    /// Two overlapping constant tiles must composite to exactly the constant — the normalized
    /// blend's defining property, and the thing the center-write probe compositor only approximated.
    @Test func constantFieldSurvivesOverlap() {
        let tile = 16, overlap = 4, w = 28, h = 16
        var colour = [Float](repeating: 0, count: 3 * w * h)
        var weight = [Float](repeating: 0, count: w * h)
        let ramp = SRVGGNetCompact_CoreAI.rampWeights(tile: tile, overlap: overlap)
        for ox in [0, 12] {
            for y in 0 ..< h {
                for x in 0 ..< tile {
                    let ww = ramp[y % tile] * ramp[x]
                    weight[y * w + (ox + x)] += ww
                    for c in 0 ..< 3 { colour[c * w * h + y * w + (ox + x)] += ww * 0.5 }
                }
            }
        }
        let img = SRVGGNetCompact_CoreAI.image(colour: colour, weight: weight, width: w, height: h)
        #expect(img.width == w && img.height == h)
        // Every covered pixel must be 128 (0.5 quantized once) — a seam would show as ±1 ripples.
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        for y in 0 ..< h { for x in 0 ..< w where weight[y * w + x] > 0 {
            #expect(rgba[(y * w + x) * 4] == 128, "seam ripple at \(x),\(y)")
        } }
    }
}

@Suite("Live ANE")
struct LiveTests {
    /// One real tiled upscale per variant through the bundled asset. ⚠️ ~8 s first-run E5RT
    /// specialization per variant per machine; skipped (not silently passed) where CoreAI is absent.
    @Test(arguments: SRVGGNetCompact_CoreAI.Variant.allCases)
    func upscaleRunsAndScales(variant: SRVGGNetCompact_CoreAI.Variant) async throws {
        let side = 200   // forces the tiled path incl. clamped edge tiles
        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        for i in 0 ..< side * side {   // deterministic gradient + checker, not a flat field
            let x = i % side, y = i / side
            pixels[i * 4] = UInt8((x * 255) / side)
            pixels[i * 4 + 1] = UInt8((y * 255) / side)
            pixels[i * 4 + 2] = ((x / 8 + y / 8) % 2 == 0) ? 200 : 55
            pixels[i * 4 + 3] = 255
        }
        let ctx = CGContext(data: &pixels, width: side, height: side, bitsPerComponent: 8,
                            bytesPerRow: side * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let input = ctx.makeImage()!

        let model = SRVGGNetCompact_CoreAI(variant: variant)
        let out = try await model.upscale(input)
        #expect(out.width == side * 4)
        #expect(out.height == side * 4)
    }
}
