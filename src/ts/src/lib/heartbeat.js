"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createStopEvent = createStopEvent;
exports.startPingLoop = startPingLoop;
const events_1 = require("events");
const grpc_js_1 = require("@grpc/grpc-js");
const retry_invoker_1 = require("./retry_invoker");
const constants_1 = require("./constants");
class StopEvent extends events_1.EventEmitter {
    is_set;
    constructor() {
        super();
        this.is_set = false;
        this.on("set", () => {
            this.is_set = true;
        });
    }
    set() {
        this.emit("set");
    }
}
function createStopEvent() {
    return new StopEvent();
}
function pingLoop(pingFn, stopEvent) {
    const waitFn = (waitTime) => new Promise((resolve) => {
        if (!stopEvent.is_set) {
            setTimeout(resolve, waitTime * 1000);
        }
    });
    const onBackoff = (state) => {
        const err = state.exception;
        if (!err)
            return;
        const statusCode = err.code;
        if (statusCode === grpc_js_1.status.DEADLINE_EXCEEDED) {
            if (state.actualWait !== undefined) {
                state.actualWait = Math.max(state.actualWait - constants_1.PING_CALL_TIMEOUT, 0);
            }
        }
    };
    const wrappedPing = () => {
        if (!stopEvent.is_set) {
            pingFn();
        }
    };
    const retrier = new retry_invoker_1.RetryInvoker(retry_invoker_1.exponential, Error, null, null, {
        onBackoff,
        waitFunction: waitFn,
    });
    return new Promise(async (resolve) => {
        while (!stopEvent.is_set) {
            await retrier.invoke(wrappedPing);
        }
        resolve(); // Resolve when stopEvent is triggered
    });
}
// TypeScript version of startPingLoop
function startPingLoop(pingFn, stopEvent) {
    // Start the loop, but do not block
    pingLoop(pingFn, stopEvent).then(() => {
        console.log("Ping loop terminated.");
    });
    const intervalId = setInterval(() => {
        if (stopEvent.is_set) {
            clearInterval(intervalId); // Clear the interval when stopEvent is set
        }
    }, 1000); // Interval to keep the loop alive
    return intervalId; // Return the interval ID}
}
