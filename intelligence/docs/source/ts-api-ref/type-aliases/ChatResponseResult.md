[TypeScript API](../index.md) / ChatResponseResult

# Type Alias: ChatResponseResult

> **ChatResponseResult**: \{ `message`: [`Message`](../interfaces/Message.md); `ok`: `true`; \} \| \{ `failure`: [`Failure`](../interfaces/Failure.md); `ok`: `false`; \}

Defined in: [typing.ts:307](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L307)

Represents the result of a chat operation.

This union type encapsulates both successful and failed chat outcomes.

- **Success:**
  When the operation is successful, the result is an object with:
  - `ok: true` indicating success.
  - `message: {@link Message}` containing the chat response.

- **Failure:**
  When the operation fails, the result is an object with:
  - `ok: false` indicating failure.
  - `failure: {@link Failure}` providing details about the error.
