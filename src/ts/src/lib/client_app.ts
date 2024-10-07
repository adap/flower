import { Context, Message, MessageType, ClientFnExt, Mod, ClientAppCallable } from "./typing";
import { Client } from "./client";
import { handleLegacyMessageFromMsgType } from "./message_handler";
import { warnDeprecatedFeature, warnPreviewFeature } from "./logger"; // Mock for warnings

export function makeFFN(ffn: ClientAppCallable, mods: Mod[]): ClientAppCallable {
  function wrapFFN(_ffn: ClientAppCallable, _mod: Mod): ClientAppCallable {
    return function newFFN(message: Message, context: Context): Message {
      return _mod(message, context, _ffn); // Call the mod with the message, context, and original ffn
    };
  }

  // Apply each mod to ffn, in reversed order
  for (const mod of mods.reverse()) {
    ffn = wrapFFN(ffn, mod);
  }

  return ffn; // Return the modified ffn
}

function alertErroneousClientFn(): void {
  throw new Error(
    "A `ClientApp` cannot make use of a `client_fn` that does not have a signature in the form: `function client_fn(context: Context)`. You can import the `Context` like this: `import { Context } from './common'`"
  );
}

function inspectMaybeAdaptClientFnSignature(clientFn: ClientFnExt): ClientFnExt {
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

export class ClientAppException extends Error {
  constructor(message: string) {
    const exName = "ClientAppException";
    super(`\nException ${exName} occurred. Message: ${message}`);
    this.name = exName;
  }
}

export class ClientApp {
  private _mods: Mod[];
  private _call: ClientAppCallable | null = null;
  private _train: ClientAppCallable | null = null;
  private _evaluate: ClientAppCallable | null = null;
  private _query: ClientAppCallable | null = null;

  constructor(clientFn?: ClientFnExt, mods?: Mod[]) {
    this._mods = mods || [];

    if (clientFn) {
      clientFn = inspectMaybeAdaptClientFnSignature(clientFn);

      const ffn: ClientAppCallable = (message, context) => {
        return handleLegacyMessageFromMsgType(clientFn!, message, context);
      };

      this._call = makeFFN(ffn, this._mods);
    }
  }

  call(message: Message, context: Context): Message {
    if (this._call) {
      return this._call(message, context);
    }

    switch (message.metadata.messageType) {
      case MessageType.TRAIN:
        if (this._train) return this._train(message, context);
        throw new Error("No `train` function registered");
      case MessageType.EVALUATE:
        if (this._evaluate) return this._evaluate(message, context);
        throw new Error("No `evaluate` function registered");
      case MessageType.QUERY:
        if (this._query) return this._query(message, context);
        throw new Error("No `query` function registered");
      default:
        throw new Error(`Unknown message_type: ${message.metadata.messageType}`);
    }
  }

  train(): (trainFn: ClientAppCallable) => ClientAppCallable {
    return (trainFn: ClientAppCallable) => {
      if (this._call) {
        throw registrationError("train");
      }

      warnPreviewFeature("ClientApp-register-train-function");
      this._train = makeFFN(trainFn, this._mods);
      return trainFn;
    };
  }

  evaluate(): (evaluateFn: ClientAppCallable) => ClientAppCallable {
    return (evaluateFn: ClientAppCallable) => {
      if (this._call) {
        throw registrationError("evaluate");
      }

      warnPreviewFeature("ClientApp-register-evaluate-function");
      this._evaluate = makeFFN(evaluateFn, this._mods);
      return evaluateFn;
    };
  }

  query(): (queryFn: ClientAppCallable) => ClientAppCallable {
    return (queryFn: ClientAppCallable) => {
      if (this._call) {
        throw registrationError("query");
      }

      warnPreviewFeature("ClientApp-register-query-function");
      this._query = makeFFN(queryFn, this._mods);
      return queryFn;
    };
  }
}

export class LoadClientAppError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LoadClientAppError";
  }
}

function registrationError(fnName: string): Error {
  return new Error(
    `Use either \`@app.${fnName}()\` or \`clientFn\`, but not both.\n\n` +
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
    `});\n\`\`\`\n`
  );
}
