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
