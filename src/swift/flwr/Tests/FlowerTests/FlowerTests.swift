import XCTest
@testable import flwr

@available(iOS 14.0, *)
final class FlowerTests: XCTestCase {
    func testParameterConverter() throws {
        let pConv = ParameterConverter.shared
        let input1: [Float] = [1, 2, 3, 4]
        let input2: [Float] = [1.1435, 2, 3, 4]
        let output1 = pConv.dataToArray(
            data: pConv.arrayToData(
                array: input1,
                shape: [2, 2]
            )!
        )
        let output2 = pConv.dataToArray(
            data: pConv.arrayToData(
                array: input2,
                shape: [2, 2]
            )!
        )
        XCTAssertEqual(input1, output1)
        XCTAssertEqual(input2, output2)
    }
}
