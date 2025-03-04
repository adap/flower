[TypeScript API](../index.md) / ToolFunctionParameters

# Interface: ToolFunctionParameters

Defined in: [typing.ts:64](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L64)

Represents the parameters required for a tool's function.

## Properties

### properties

> **properties**: `Record`\<`string`, [`ToolParameterProperty`](ToolParameterProperty.md)\>

Defined in: [typing.ts:73](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L73)

A record defining the properties of each parameter.

***

### required

> **required**: `string`[]

Defined in: [typing.ts:78](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L78)

A list of parameter names that are required.

***

### type

> **type**: `string`

Defined in: [typing.ts:68](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L68)

The data type of the parameters (e.g., "object").
