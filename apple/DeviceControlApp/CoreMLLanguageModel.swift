import Foundation
import CoreML
import DeviceControlKit

/// On-device SLM wrapper: runs the exported Core ML model (DCSLM.mlpackage) and
/// greedy-decodes the command DSL char-by-char. Conforms to DeviceControlKit's
/// `LanguageModel`, so it drops straight into `DeviceAgent`.
///
/// Expects `DCSLM.mlmodelc` and `dcslm_tokenizer.json` in the app bundle
/// (produce them with tools/export_coreml.py).
public final class CoreMLLanguageModel: LanguageModel {
    private let model: MLModel
    private let stoi: [String: Int]
    private let itos: [Int: String]
    private let block: Int
    private let U = "\u{01}", C = "\u{02}", E = "\u{03}"

    public init() throws {
        guard let modelURL = Bundle.main.url(forResource: "DCSLM", withExtension: "mlmodelc"),
              let tokURL = Bundle.main.url(forResource: "dcslm_tokenizer", withExtension: "json")
        else { throw NSError(domain: "DCSLM", code: 1,
                             userInfo: [NSLocalizedDescriptionKey: "model/tokenizer not bundled"]) }
        self.model = try MLModel(contentsOf: modelURL)
        let tok = try JSONSerialization.jsonObject(with: Data(contentsOf: tokURL)) as! [String: Any]
        self.stoi = tok["stoi"] as! [String: Int]
        self.block = tok["block_size"] as! Int
        var itos: [Int: String] = [:]
        for (k, v) in (tok["itos"] as! [String: String]) { itos[Int(k)!] = v }
        self.itos = itos
    }

    public func plan(_ request: String) async throws -> String {
        var ids = (U + request + C).compactMap { stoi[String($0)] }
        let endID = stoi[E]
        var out = ""
        for _ in 0..<96 {
            let window = Array(ids.suffix(block))
            let tokens = try MLMultiArray(shape: [1, NSNumber(value: block)], dataType: .int32)
            for i in 0..<block {
                tokens[[0, NSNumber(value: i)]] = NSNumber(value: i < window.count ? window[i] : 0)
            }
            let input = try MLDictionaryFeatureProvider(dictionary: ["tokens": MLFeatureValue(multiArray: tokens)])
            let logits = try model.prediction(from: input).featureValue(for: "logits")!.multiArrayValue!
            let pos = min(window.count, block) - 1
            let vocab = logits.shape[2].intValue
            var best = 0; var bestVal = -Double.infinity
            for v in 0..<vocab {
                let val = logits[[0, NSNumber(value: pos), NSNumber(value: v)]].doubleValue
                if val > bestVal { bestVal = val; best = v }
            }
            if best == endID { break }
            out.append(itos[best] ?? "")
            ids.append(best)
        }
        return out
    }
}
