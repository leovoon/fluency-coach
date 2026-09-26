// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "FluidPoc",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(
            url: "https://github.com/FluidInference/FluidAudio",
            revision: "20d4f0bd46d11d7f50a6eb4f7835cfdbd2b4ba14"
        )
    ],
    targets: [
        .executableTarget(name: "fluidpoc", dependencies: ["FluidAudio"])
    ]
)
