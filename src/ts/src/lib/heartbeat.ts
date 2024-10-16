import { EventEmitter } from "events";
import { ServiceError, status } from "@grpc/grpc-js";
import { RetryInvoker, RetryState, exponential } from "./retry_invoker";
import { PING_CALL_TIMEOUT } from "./constants";


class StopEvent extends EventEmitter {
  public is_set: boolean;

  constructor() {
    super();
    this.is_set = false;

    this.on("set", () => {
      this.is_set = true;
    });
  }

  set(): void {
    this.emit("set");
  }
}

export function createStopEvent(): StopEvent {
  return new StopEvent();
}

function pingLoop(pingFn: () => void, stopEvent: StopEvent): Promise<void> {
  const waitFn = (waitTime: number): Promise<void> =>
    new Promise((resolve) => {
      if (!stopEvent.is_set) {
        setTimeout(resolve, waitTime * 1000);
      }
    });

  const onBackoff = (state: RetryState): void => {
    const err = state.exception as ServiceError;
    if (!err) return;

    const statusCode = err.code;
    if (statusCode === status.DEADLINE_EXCEEDED) {
      if (state.actualWait !== undefined) {
        state.actualWait = Math.max(state.actualWait - PING_CALL_TIMEOUT, 0);
      }
    }
  };

  const wrappedPing = (): void => {
    if (!stopEvent.is_set) {
      pingFn();
    }
  };

  const retrier = new RetryInvoker(exponential, Error, null, null, {
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
export function startPingLoop(pingFn: () => void, stopEvent: StopEvent): NodeJS.Timeout {
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
