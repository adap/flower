"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LoadClientAppError = exports.ClientApp = exports.ClientAppException = void 0;
exports.makeFFN = makeFFN;
const typing_1 = require("./typing");
const message_handler_1 = require("./message_handler");
const logger_1 = require("./logger"); // Mock for warnings
function makeFFN(ffn, mods) {
    function wrapFFN(_ffn, _mod) {
        return function newFFN(message, context) {
            return _mod(message, context, _ffn); // Call the mod with the message, context, and original ffn
        };
    }
    // Apply each mod to ffn, in reversed order
    for (const mod of mods.reverse()) {
        ffn = wrapFFN(ffn, mod);
    }
    return ffn; // Return the modified ffn
}
function alertErroneousClientFn() {
    throw new Error("A `ClientApp` cannot make use of a `client_fn` that does not have a signature in the form: `function client_fn(context: Context)`. You can import the `Context` like this: `import { Context } from './common'`");
}
function inspectMaybeAdaptClientFnSignature(clientFn) {
    if (clientFn.length !== 1) {
        alertErroneousClientFn();
    }
    // const firstArg = clientFn.arguments[0];
    // if (typeof firstArg === "string") {
    //   warnDeprecatedFeature(
    //     "`clientFn` now expects a signature `function clientFn(context: Context)`. The provided `clientFn` has a signature `function clientFn(cid: string)`"
    //   );
    //   return (context: Context): Client => {
    //     const cid = context.nodeConfig["partition-id"] || context.nodeId;
    //     return clientFn(cid as any);
    //   };
    // }
    return clientFn;
}
class ClientAppException extends Error {
    constructor(message) {
        const exName = "ClientAppException";
        super(`\nException ${exName} occurred. Message: ${message}`);
        this.name = exName;
    }
}
exports.ClientAppException = ClientAppException;
class ClientApp {
    _mods;
    _call = null;
    _train = null;
    _evaluate = null;
    _query = null;
    constructor(clientFn, mods) {
        this._mods = mods || [];
        if (clientFn) {
            clientFn = inspectMaybeAdaptClientFnSignature(clientFn);
            const ffn = (message, context) => {
                return (0, message_handler_1.handleLegacyMessageFromMsgType)(clientFn, message, context);
            };
            this._call = makeFFN(ffn, this._mods);
        }
    }
    call(message, context) {
        if (this._call) {
            return this._call(message, context);
        }
        switch (message.metadata.messageType) {
            case typing_1.MessageType.TRAIN:
                if (this._train)
                    return this._train(message, context);
                throw new Error("No `train` function registered");
            case typing_1.MessageType.EVALUATE:
                if (this._evaluate)
                    return this._evaluate(message, context);
                throw new Error("No `evaluate` function registered");
            case typing_1.MessageType.QUERY:
                if (this._query)
                    return this._query(message, context);
                throw new Error("No `query` function registered");
            default:
                throw new Error(`Unknown message_type: ${message.metadata.messageType}`);
        }
    }
    train() {
        return (trainFn) => {
            if (this._call) {
                throw registrationError("train");
            }
            (0, logger_1.warnPreviewFeature)("ClientApp-register-train-function");
            this._train = makeFFN(trainFn, this._mods);
            return trainFn;
        };
    }
    evaluate() {
        return (evaluateFn) => {
            if (this._call) {
                throw registrationError("evaluate");
            }
            (0, logger_1.warnPreviewFeature)("ClientApp-register-evaluate-function");
            this._evaluate = makeFFN(evaluateFn, this._mods);
            return evaluateFn;
        };
    }
    query() {
        return (queryFn) => {
            if (this._call) {
                throw registrationError("query");
            }
            (0, logger_1.warnPreviewFeature)("ClientApp-register-query-function");
            this._query = makeFFN(queryFn, this._mods);
            return queryFn;
        };
    }
}
exports.ClientApp = ClientApp;
class LoadClientAppError extends Error {
    constructor(message) {
        super(message);
        this.name = "LoadClientAppError";
    }
}
exports.LoadClientAppError = LoadClientAppError;
function registrationError(fnName) {
    return new Error(`Use either \`@app.${fnName}()\` or \`clientFn\`, but not both.\n\n` +
        `Use the \`ClientApp\` with an existing \`clientFn\`:\n\n` +
        `\`\`\`\nclass FlowerClient extends NumPyClient {}\n\n` +
        `function clientFn(context: Context) {\n` +
        `  return new FlowerClient().toClient();\n` +
        `}\n\n` +
        `const app = new ClientApp({ clientFn });\n\`\`\`\n\n` +
        `Use the \`ClientApp\` with a custom ${fnName} function:\n\n` +
        `\`\`\`\nconst app = new ClientApp();\n\n` +
        `app.${fnName}((message, context) => {\n` +
        `  console.log("ClientApp ${fnName} running");\n` +
        `  return message.createReply({ content: message.content });\n` +
        `});\n\`\`\`\n`);
}
