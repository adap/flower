"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const retry_invoker_1 = require("./retry_invoker"); // Adjust import paths as necessary
describe("RetryInvoker", () => {
    // Mocking utilities (similar to pytest's fixture)
    let mockTime;
    let mockSleep;
    beforeEach(() => {
        mockTime = jest.spyOn(Date, "now").mockImplementation(() => 0);
        mockSleep = jest
            .spyOn(global, "setTimeout")
            .mockImplementation((fn, ms) => {
            fn(); // Immediately call the function to avoid waiting
            return setTimeout(fn, ms); // Return a proper Timeout object
        });
    });
    afterEach(() => {
        mockTime.mockRestore();
        mockSleep.mockRestore();
    });
    // Test successful invocation
    it("should succeed when the function does not throw", async () => {
        // Prepare
        const successHandler = jest.fn();
        const backoffHandler = jest.fn();
        const giveupHandler = jest.fn();
        const invoker = new retry_invoker_1.RetryInvoker(() => (0, retry_invoker_1.constant)(0.1), Error, null, null, {
            onSuccess: successHandler,
            onBackoff: backoffHandler,
            onGiveup: giveupHandler,
        });
        const successfulFunction = () => "success";
        // Assert that the invoker returns the correct result
        const result = await invoker.invoke(successfulFunction);
        expect(result).toBe("success");
    });
    it("should retry and fail on the failing function", async () => {
        const invoker = new retry_invoker_1.RetryInvoker(() => (0, retry_invoker_1.constant)(0.1), // Retry every 0.1 seconds
        Error, // Retry on Error
        2, // Maximum 2 retries
        2.5);
        const failingFunction = () => {
            throw new Error("failed");
        };
        // Assert that the invoker throws an error
        await expect(async () => {
            await invoker.invoke(failingFunction);
        }).rejects.toThrow("failed");
    });
    it("should call onSuccess handler when successful", async () => {
        const successHandler = jest.fn();
        const invoker = new retry_invoker_1.RetryInvoker(() => (0, retry_invoker_1.constant)(0.1), Error, 2, 2.5, {
            onSuccess: successHandler,
        });
        const successfulFunction = () => "success";
        // Call the function and assert success
        await invoker.invoke(successfulFunction);
        // Ensure the onSuccess handler was called
        expect(successHandler).toHaveBeenCalled();
    });
    it("should retry on failure and call onBackoff", async () => {
        const backoffHandler = jest.fn();
        const invoker = new retry_invoker_1.RetryInvoker(() => (0, retry_invoker_1.constant)(0.1), Error, 2, 2.5, {
            onBackoff: backoffHandler,
        });
        const failingFunction = () => {
            throw new Error("failed");
        };
        // Assert the invoker throws an error and triggers retries
        await expect(invoker.invoke(failingFunction)).rejects.toThrow("failed");
        // Ensure the backoff handler was called
        expect(backoffHandler).toHaveBeenCalled();
    });
    it("should stop after max retries", async () => {
        const invoker = new retry_invoker_1.RetryInvoker(() => (0, retry_invoker_1.constant)(0.1), Error, 2, // Max retries is 2
        2.5);
        const failingFunction = () => {
            throw new Error("failed");
        };
        // Assert the invoker gives up after max retries
        await expect(invoker.invoke(failingFunction)).rejects.toThrow("failed");
    });
});
