[TypeScript API](../index.md) / Message

# Interface: Message

Defined in: [typing.ts:6](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L6)

Represents a message in a chat session.

## Properties

### content

> **content**: `string`

Defined in: [typing.ts:15](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L15)

The content of the message.

***

### role

> **role**: `string`

Defined in: [typing.ts:10](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L10)

The role of the sender (e.g., "user", "system", "assistant").

***

### toolCalls?

> `optional` **toolCalls**: [`ToolCall`](../type-aliases/ToolCall.md)[]

Defined in: [typing.ts:20](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L20)

An optional list of calls to specific tools
