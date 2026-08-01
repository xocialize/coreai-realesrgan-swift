// swift-tools-version: 6.2

// coreai-realesrgan-swift — Real-ESRGAN (SRVGGNetCompact) on the Apple Neural Engine via CoreAI.
//
// The first `coreai-*` package, sibling to `mlx-realesrgan-swift` (same capability home, different
// backend — the naming convention decided 2026-07-31). Why it exists, measured (`GAP-PROGRAM.md`
// V13): ANE wall-clock TIES the best MLX-GPU configuration at t128 while drawing **≈4.5–4.9× less
// energy per frame** (~17 W vs ~83 W over idle). For long batch/video renders on a laptop, power and
// thermals are the product axis, and this is the only backend that moves them.
//
// ⚠️ macOS 27+ ONLY (CoreAI.framework). The MLX sibling is the fallback below 27 — a consuming host
// selects per OS; this package does not pretend to run where its framework does not exist.
//
// 🔑 Tile geometry is FIXED at 128/8, the inversion of the MLX sibling's injectable geometry and
// deliberate: the ANE compiles one executable per static shape, so a tile size is a build-time
// artifact here, not a runtime knob. 128 is the measured optimum on BOTH backends (V13 ladder).
import PackageDescription

let package = Package(
    name: "coreai-realesrgan-swift",
    platforms: [.macOS("27.0")],
    products: [
        .library(name: "RealESRGANCoreAI", targets: ["RealESRGANCoreAI"]),
    ],
    targets: [
        .target(
            name: "RealESRGANCoreAI",
            resources: [
                .copy("Resources/realesr_general_x4v3_float16_static128.aimodel"),
                .copy("Resources/realesr_general_wdn_x4v3_float16_static128.aimodel"),
                .copy("Resources/realesr_animevideov3_float16_static128.aimodel"),
            ],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .testTarget(name: "RealESRGANCoreAITests", dependencies: ["RealESRGANCoreAI"]),
    ]
)
