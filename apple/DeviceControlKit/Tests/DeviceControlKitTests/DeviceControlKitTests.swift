import XCTest
@testable import DeviceControlKit

final class DeviceControlKitTests: XCTestCase {

    func testParseSingle() {
        guard case .actions(let cmds) = DeviceControlEngine.parse("set_wifi|enabled=false") else {
            return XCTFail("expected actions")
        }
        XCTAssertEqual(cmds, [DeviceCommand(tool: "set_wifi", args: ["enabled": "false"])])
    }

    func testParseCompositional() {
        guard case .actions(let cmds) =
            DeviceControlEngine.parse("set_wifi|enabled=false ; set_brightness|level=30") else {
            return XCTFail("expected actions")
        }
        XCTAssertEqual(cmds.count, 2)
        XCTAssertEqual(cmds[1], DeviceCommand(tool: "set_brightness", args: ["level": "30"]))
    }

    func testParseClarifyAndRefuse() {
        XCTAssertEqual(DeviceControlEngine.parse("ASK"), .clarify)
        XCTAssertEqual(DeviceControlEngine.parse("REFUSE"), .refuse)
    }

    func testRecipeiOSUsesShortcutForWifi() {
        // No direct Wi-Fi API on iOS -> the recipe must fall back to a Shortcuts action.
        XCTAssertEqual(Recipe.plan("set_wifi", on: .iOS), [.shortcut("Set Wi-Fi")])
    }

    func testRecipeiOSBrightnessIsDirect() {
        XCTAssertEqual(Recipe.plan("set_brightness", on: .iOS).first, .directAPI)
    }

    func testRecipeMacHasAccessibilityFallback() {
        XCTAssertEqual(Recipe.plan("set_wifi", on: .macOS), [.directAPI, .accessibility])
    }

    func testConfirmationGateBlocksSensitive() async {
        let results = await DeviceControlEngine.execute(
            dsl: "send_message|to=boss|body=I quit", platform: .iOS, confirm: { _ in false })
        XCTAssertEqual(results.count, 1)
        XCTAssertFalse(results[0].ok)          // gate refused -> not executed
    }
}
