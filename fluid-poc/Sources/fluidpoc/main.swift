// POC: render text with a cloned teaching voice via FluidAudio PocketTTS.
//
// Single-shot:  fluidpoc <reference.wav> "<text>" <out.wav> [temperature]
// Worker mode:  fluidpoc --worker <reference.wav>
//   stdin:  {"text": "...", "out": "/abs/file.wav"}
//   stdout: WORKER_READY, then one OK / ERROR line per request.
// First run downloads ~766MB of CoreML models (English pack) + the
// language-agnostic mimi encoder used for voice cloning.

import Foundation
import FluidAudio

func stderrLog(_ s: String) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
}

@main
struct Main {
    static func main() async throws {
        setbuf(stdout, nil)
        let args = CommandLine.arguments

        if args.count >= 2 && args[1] == "--worker" {
            guard args.count >= 3 else {
                stderrLog("worker mode needs <reference.wav>")
                exit(2)
            }
            try await worker(refURL: URL(fileURLWithPath: args[2]))
            return
        }

        guard args.count >= 4 else {
            FileHandle.standardError.write(Data("usage: fluidpoc <ref.wav> \"<text>\" <out.wav> [temperature]\n".utf8))
            exit(2)
        }
        let refURL = URL(fileURLWithPath: args[1])
        let text = args[2]
        let outURL = URL(fileURLWithPath: args[3])
        let temperature: Float = args.count > 4 ? Float(args[4]) ?? 0.7 : 0.7

        let t0 = Date()
        let manager = PocketTtsManager()
        try await manager.initialize()
        stderrLog("initialize: \(Date().timeIntervalSince(t0))s")
        let tc = Date()
        let voice = try await manager.cloneVoice(from: refURL)
        stderrLog("cloneVoice: \(Date().timeIntervalSince(tc))s")
        let ts = Date()
        let audio = try await manager.synthesize(
            text: text,
            voiceData: voice,
            temperature: temperature
        )
        stderrLog("synthesize: \(Date().timeIntervalSince(ts))s")
        stderrLog("total: \(Date().timeIntervalSince(t0))s")
        try audio.write(to: outURL)
        print("OK \(outURL.path)")
    }

    static func worker(refURL: URL) async throws {
        let manager = PocketTtsManager()
        try await manager.initialize()
        let voice = try await manager.cloneVoice(from: refURL)
        stderrLog("worker ready (ref: \(refURL.lastPathComponent))")
        print("WORKER_READY")
        while let line = readLine() {
            guard let data = line.data(using: .utf8),
                  let val = try? JSONSerialization.jsonObject(with: data) as? [String: String],
                  let text = val["text"], let out = val["out"],
                  !text.isEmpty, !out.isEmpty else {
                print("ERROR bad request")
                continue
            }
            do {
                let audio = try await manager.synthesize(
                    text: text,
                    voiceData: voice,
                    temperature: 0.7
                )
                try audio.write(to: URL(fileURLWithPath: out))
                print("OK")
            } catch {
                print("ERROR \(error)")
            }
        }
    }
}
