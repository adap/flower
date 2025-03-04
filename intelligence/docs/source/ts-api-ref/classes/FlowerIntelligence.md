[TypeScript API](../index.md) / FlowerIntelligence

# Class: FlowerIntelligence

Defined in: [flowerintelligence.ts:28](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/flowerintelligence.ts#L28)

Class representing the core intelligence service for Flower Labs.
It facilitates chat, generation, and summarization tasks, with the option of using a
local or remote engine based on configurations and availability.

## Methods

### chat()

Conducts a chat interaction using the specified model and options.

This method can be invoked in one of two ways:

1. With a string input (plus an optional options object). In this case the string
   is automatically wrapped as a single message with role 'user'.

   Example:
   ```ts
   fi.chat("Why is the sky blue?", { temperature: 0.7 });
   ```

2. With a single object that includes a [Message](../interfaces/Message.md) array along with additional options.

   Example:
   ```ts
   fi.chat({
     messages: [{ role: 'user', content: "Why is the sky blue?" }],
     model: "meta/llama3.2-1b"
   });
   ```

#### Param

Either a string input or a [ChatOptions](../interfaces/ChatOptions.md) object that must include a `messages` array.

#### Param

An optional [ChatOptions](../interfaces/ChatOptions.md) object (used only when the first parameter is a string).

#### Call Signature

> **chat**(`input`, `options`?): `Promise`\<[`ChatResponseResult`](../type-aliases/ChatResponseResult.md)\>

Defined in: [flowerintelligence.ts:92](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/flowerintelligence.ts#L92)

##### Parameters

###### input

`string`

###### options?

[`ChatOptions`](../interfaces/ChatOptions.md)

##### Returns

`Promise`\<[`ChatResponseResult`](../type-aliases/ChatResponseResult.md)\>

#### Call Signature

> **chat**(`options`): `Promise`\<[`ChatResponseResult`](../type-aliases/ChatResponseResult.md)\>

Defined in: [flowerintelligence.ts:95](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/flowerintelligence.ts#L95)

##### Parameters

###### options

[`ChatOptions`](../interfaces/ChatOptions.md) & `object`

##### Returns

`Promise`\<[`ChatResponseResult`](../type-aliases/ChatResponseResult.md)\>

***

### fetchModel()

> **fetchModel**(`model`, `callback`): `Promise`\<[`Result`](../type-aliases/Result.md)\<`void`\>\>

Defined in: [flowerintelligence.ts:81](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/flowerintelligence.ts#L81)

Downloads and loads a model into memory.

#### Parameters

##### model

`string`

Model name to use for the chat.

##### callback

(`progress`) => `void`

A callback to handle the progress of the download

#### Returns

`Promise`\<[`Result`](../type-aliases/Result.md)\<`void`\>\>

A [Result](../type-aliases/Result.md) containing either a [Failure](../interfaces/Failure.md) (containing `code: number` and `description: string`) if `ok` is false or a value of `void`, if `ok` is true (meaning the loading was successful).

## Accessors

### apiKey

#### Set Signature

> **set** **apiKey**(`apiKey`): `void`

Defined in: [flowerintelligence.ts:71](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/flowerintelligence.ts#L71)

Set apiKey for FlowerIntelligence.

##### Parameters

###### apiKey

`string`

##### Returns

`void`

***

### remoteHandoff

#### Get Signature

> **get** **remoteHandoff**(): `boolean`

Defined in: [flowerintelligence.ts:64](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/flowerintelligence.ts#L64)

Gets the current remote handoff status.

##### Returns

`boolean`

boolean - the value of the remote handoff variable

#### Set Signature

> **set** **remoteHandoff**(`remoteHandoffValue`): `void`

Defined in: [flowerintelligence.ts:56](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/flowerintelligence.ts#L56)

Sets the remote handoff boolean.

##### Parameters

###### remoteHandoffValue

`boolean`

If true, the processing might be done on a secure
remote server instead of locally (if resources are lacking).

##### Returns

`void`

***

### instance

#### Get Signature

> **get** `static` **instance**(): [`FlowerIntelligence`](FlowerIntelligence.md)

Defined in: [flowerintelligence.ts:45](https://github.com/adap/flower/blob/0a8a2219007e2bbfc1082df3392f666e281d1516/intelligence/ts/src/flowerintelligence.ts#L45)

Get the initialized FlowerIntelligence instance.
Initializes the instance if it doesn't exist.

##### Returns

[`FlowerIntelligence`](FlowerIntelligence.md)

The initialized FlowerIntelligence instance.

## Constructors

### new FlowerIntelligence()

> **new FlowerIntelligence**(): [`FlowerIntelligence`](FlowerIntelligence.md)

#### Returns

[`FlowerIntelligence`](FlowerIntelligence.md)
