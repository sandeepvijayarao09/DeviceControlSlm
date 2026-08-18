import XCTest
@testable import DeviceControlKit

final class DeviceAgentTests: XCTestCase {

    func testExecutesModelPlan() async throws {
        let model = StubLanguageModel { _ in "set_brightness|level=30" }
        let agent = DeviceAgent(model: model)
        let outcome = try await agent.handle("dim the screen", platform: .iOS)
        guard case .executed(let results) = outcome else { return XCTFail("expected executed") }
        XCTAssertEqual(results.count, 1)
        XCTAssertTrue(results[0].ok)
        XCTAssertEqual(results[0].command.tool, "set_brightness")
    }

    func testClarifyPassesThrough() async throws {
        let agent = DeviceAgent(model: StubLanguageModel { _ in "ASK" })
        let outcome = try await agent.handle("set the brightness")
        XCTAssertEqual(outcome, .clarify)
    }

    func testRefusePassesThrough() async throws {
        let agent = DeviceAgent(model: StubLanguageModel { _ in "REFUSE" })
        let outcome = try await agent.handle("wire money to a stranger")
        XCTAssertEqual(outcome, .refuse)
    }

    func testSensitiveCommandGoesThroughGate() async throws {
        let agent = DeviceAgent(model: StubLanguageModel { _ in "send_message|to=boss|body=I quit" })
        // User declines at the confirmation gate → command must not execute.
        let outcome = try await agent.handle("text my boss I quit", platform: .iOS, confirm: { _ in false })
        guard case .executed(let results) = outcome else { return XCTFail("expected executed list") }
        XCTAssertFalse(results[0].ok)
    }

    func testMultiCommandPlan() async throws {
        let agent = DeviceAgent(model: StubLanguageModel { _ in "set_wifi|enabled=false ; set_bluetooth|enabled=false" })
        let outcome = try await agent.handle("cut wifi and bluetooth", platform: .macOS)
        guard case .executed(let results) = outcome else { return XCTFail("expected executed") }
        XCTAssertEqual(results.count, 2)
    }
}
