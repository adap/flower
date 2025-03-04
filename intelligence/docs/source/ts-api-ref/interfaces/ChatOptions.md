[TypeScript API](../index.md) / ChatOptions

# Interface: ChatOptions

Defined in: [typing.ts:241](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/typing.ts#L241)

Options to configure the chat interaction.

## Properties

### encrypt?

> `optional` **encrypt**: `boolean`

Defined in: [typing.ts:286](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/typing.ts#L286)

If true, enables end-to-end encryption for processing the request.

***

### forceLocal?

> `optional` **forceLocal**: `boolean`

Defined in: [typing.ts:281](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/typing.ts#L281)

If true, forces the use of a local engine.

***

### forceRemote?

> `optional` **forceRemote**: `boolean`

Defined in: [typing.ts:276](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/typing.ts#L276)

If true and remote handoff is enabled, forces the use of a remote engine.

***

### maxCompletionTokens?

> `optional` **maxCompletionTokens**: `number`

Defined in: [typing.ts:255](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/typing.ts#L255)

Maximum number of tokens to generate in the response.

***

### model?

> `optional` **model**: `string`

Defined in: [typing.ts:245](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/typing.ts#L245)

The model name to use for the chat. Defaults to a predefined model if not specified.

***

### onStreamEvent()?

> `optional` **onStreamEvent**: (`event`) => `void`

Defined in: [typing.ts:266](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/typing.ts#L266)

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

Defined in: [typing.ts:260](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/typing.ts#L260)

If true, the response will be streamed.

***

### temperature?

> `optional` **temperature**: `number`

Defined in: [typing.ts:250](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/typing.ts#L250)

Controls the creativity of the response. Typically a value between 0 and 1.

***

### tools?

> `optional` **tools**: [`Tool`](Tool.md)[]

Defined in: [typing.ts:271](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/typing.ts#L271)

Optional array of tools available for the chat.
