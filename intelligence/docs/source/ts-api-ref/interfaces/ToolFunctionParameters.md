[TypeScript API](../index.md) / ToolFunctionParameters

# Interface: ToolFunctionParameters

Defined in: [typing.ts:64](https://github.com/adap/flower/blob/0f847b5db7209b5c41b08d1c3aa630bfc89621fb/intelligence/ts/src/typing.ts#L64)

Represents the parameters required for a tool's function.

## Properties

### properties

> **properties**: `Record`\<`string`, [`ToolParameterProperty`](ToolParameterProperty.md)\>

Defined in: [typing.ts:73](https://github.com/adap/flower/blob/0f847b5db7209b5c41b08d1c3aa630bfc89621fb/intelligence/ts/src/typing.ts#L73)

A record defining the properties of each parameter.

***

### required

> **required**: `string`[]

Defined in: [typing.ts:78](https://github.com/adap/flower/blob/0f847b5db7209b5c41b08d1c3aa630bfc89621fb/intelligence/ts/src/typing.ts#L78)

A list of parameter names that are required.

***

### type

> **type**: `string`

Defined in: [typing.ts:68](https://github.com/adap/flower/blob/0f847b5db7209b5c41b08d1c3aa630bfc89621fb/intelligence/ts/src/typing.ts#L68)

The data type of the parameters (e.g., "object").
