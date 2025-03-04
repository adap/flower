[TypeScript API](../index.md) / ToolFunction

# Interface: ToolFunction

Defined in: [typing.ts:84](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L84)

Represents the function provided by a tool.

## Properties

### description

> **description**: `string`

Defined in: [typing.ts:93](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L93)

A brief description of what the function does.

***

### name

> **name**: `string`

Defined in: [typing.ts:88](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L88)

The name of the function provided by the tool.

***

### parameters

> **parameters**: [`ToolFunctionParameters`](ToolFunctionParameters.md)

Defined in: [typing.ts:98](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L98)

The parameters required for invoking the function.
