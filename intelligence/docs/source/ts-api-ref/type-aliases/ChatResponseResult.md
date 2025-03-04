[TypeScript API](../index.md) / ChatResponseResult

# Type Alias: ChatResponseResult

> **ChatResponseResult**: \{ `message`: [`Message`](../interfaces/Message.md); `ok`: `true`; \} \| \{ `failure`: [`Failure`](../interfaces/Failure.md); `ok`: `false`; \}

Defined in: [typing.ts:306](https://github.com/adap/flower/blob/0f847b5db7209b5c41b08d1c3aa630bfc89621fb/intelligence/ts/src/typing.ts#L306)

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
