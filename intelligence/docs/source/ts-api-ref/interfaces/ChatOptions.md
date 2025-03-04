[TypeScript API](../index.md) / ChatOptions

# Interface: ChatOptions

Defined in: [typing.ts:242](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L242)

Options to configure the chat interaction.

## Properties

### encrypt?

> `optional` **encrypt**: `boolean`

Defined in: [typing.ts:287](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L287)

If true, enables end-to-end encryption for processing the request.

***

### forceLocal?

> `optional` **forceLocal**: `boolean`

Defined in: [typing.ts:282](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L282)

If true, forces the use of a local engine.

***

### forceRemote?

> `optional` **forceRemote**: `boolean`

Defined in: [typing.ts:277](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L277)

If true and remote handoff is enabled, forces the use of a remote engine.

***

### maxCompletionTokens?

> `optional` **maxCompletionTokens**: `number`

Defined in: [typing.ts:256](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L256)

Maximum number of tokens to generate in the response.

***

### model?

> `optional` **model**: `string`

Defined in: [typing.ts:246](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L246)

The model name to use for the chat. Defaults to a predefined model if not specified.

***

### onStreamEvent()?

> `optional` **onStreamEvent**: (`event`) => `void`

Defined in: [typing.ts:267](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L267)

Optional callback invoked when a stream event occurs.

#### Parameters

##### event

[`StreamEvent`](StreamEvent.md)

The [StreamEvent](StreamEvent.md) data from the stream.

#### Returns

`void`

***

### stream?

> `optional` **stream**: `boolean`

Defined in: [typing.ts:261](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L261)

If true, the response will be streamed.

***

### temperature?

> `optional` **temperature**: `number`

Defined in: [typing.ts:251](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L251)

Controls the creativity of the response. Typically a value between 0 and 1.

***

### tools?

> `optional` **tools**: [`Tool`](Tool.md)[]

Defined in: [typing.ts:272](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L272)

Optional array of tools available for the chat.
