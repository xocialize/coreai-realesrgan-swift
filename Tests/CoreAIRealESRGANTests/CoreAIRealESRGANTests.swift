// Contract-shape tests run anywhere; the live engine-path test needs the ANE and skips honestly
// where CoreAI is absent.
import CoreGraphics
import Foundation
import MLXToolKit
import Testing

@testable import CoreAIRealESRGAN

@Suite("Manifest and configuration")
struct ManifestTests {
    @Test func manifestIsImageUpscaleAndPermissive() {
        let m = CoreAIRealESRGANUpscalePackage.manifest
        #expect(m.license.weightLicense == .bsd3)
        #expect(m.license.portCodeLicense == .mit)
        #expect(m.requirements.os.minMacOS == SemanticVersion(major: 27, minor: 0, patch: 0))
        #expect(m.requirements.requiredBackends == [.coreMLANE])
        #expect(m.requirements.footprints.map(\.quant) == [.fp16])
        #expect(m.surfaces.count == 1)
    }

    @Test func registrationConstructs() {
        _ = CoreAIRealESRGANUpscalePackage.registration
    }

    @Test func configurationRoundTrips() throws {
        let c = CoreAIRealESRGANConfiguration(variant: .anime, compute: .gpu)
        let back = try JSONDecoder().decode(CoreAIRealESRGANConfiguration.self,
                                            from: JSONEncoder().encode(c))
        #expect(back.variant == .anime)
        #expect(back.compute == .gpu)
    }
}

@Suite("Sub-native scale")
struct SubNativeResizeTests {
    /// AB-T-0232 (the AB-T-0225 fix, here): the sub-native reduction stays registered across the whole frame. 2×2
    /// white blocks on black near both corners and at the centre must stay centred on their output pixel. The
    /// CoreGraphics `.high` draw this replaced drifted ±0.26 px toward the edges (AB-R-0502).
    @Test func resizeStaysRegisteredAcrossTheFrame() throws {
        let n = 512, spots = [8, 256, 496]
        var rgba = [UInt8](repeating: 0, count: n * n * 4)
        for y in 0..<n {
            for x in 0..<n {
                let white = spots.contains { ($0..<($0 + 2)).contains(x) && ($0..<($0 + 2)).contains(y) }
                for c in 0..<4 where c == 3 || white { rgba[(y * n + x) * 4 + c] = 255 }
            }
        }
        let provider = try #require(CGDataProvider(data: Data(rgba) as CFData))
        let cg = try #require(CGImage(width: n, height: n, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: n * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                      provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let scaled = try CoreAIRealESRGANUpscalePackage.resize(cg, width: n / 2, height: n / 2)
        #expect(scaled.width == n / 2 && scaled.height == n / 2)
        // Read it back in a known RGBA8 layout (a same-size draw: no resampling).
        let m = n / 2
        var out = [UInt8](repeating: 0, count: m * m * 4)
        out.withUnsafeMutableBytes { buf in
            let ctx = CGContext(data: buf.baseAddress, width: m, height: m, bitsPerComponent: 8, bytesPerRow: m * 4,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            ctx?.draw(scaled, in: CGRect(x: 0, y: 0, width: m, height: m))
        }
        for p in spots {
            let c = p / 2
            var sum = 0.0, sx = 0.0, sy = 0.0
            for y in (c - 4)...(c + 4) {
                for x in (c - 4)...(c + 4) {
                    let v = Double(out[(y * m + x) * 4]); sum += v; sx += v * Double(x); sy += v * Double(y)
                }
            }
            #expect(abs(sx / sum - Double(c)) < 0.02 && abs(sy / sum - Double(c)) < 0.02,
                    "block at \(p): centroid (\(sx / sum), \(sy / sum)), want (\(c), \(c))")
        }
    }
}

@Suite("Engine path, live")
struct LivePackageTests {
    /// The full contract: decode → tiled ANE upscale → sub-native downsample honored →
    /// appliedScale reported. Mirrors the sibling's behavioural contract so a planner can treat
    /// the backends as interchangeable.
    @Test func subNativeScaleIsHonored() async throws {
        let side = 160
        var px = [UInt8](repeating: 0, count: side * side * 4)
        for i in 0 ..< side * side { px[i * 4] = UInt8(i % 251); px[i * 4 + 3] = 255 }
        let ctx = CGContext(data: &px, width: side, height: side, bitsPerComponent: 8,
                            bytesPerRow: side * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let png = CoreAIRealESRGANUpscalePackage.encodePNG(ctx.makeImage()!)!

        let pkg = CoreAIRealESRGANUpscalePackage(configuration: .init())
        try await pkg.load()
        let native = try #require(try await pkg.run(
            ImageUpscaleRequest(image: Image(format: .png, data: png))) as? ImageUpscaleResponse)
        #expect(native.appliedScale == 4)
        #expect(native.image.width == side * 4)

        let sub = try #require(try await pkg.run(
            ImageUpscaleRequest(image: Image(format: .png, data: png), scale: 2)) as? ImageUpscaleResponse)
        #expect(sub.appliedScale == 2)
        #expect(sub.image.width == side * 2)
        await pkg.unload()
    }
}
