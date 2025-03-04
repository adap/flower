[TypeScript API](../index.md) / FailureCode

# Enumeration: FailureCode

Defined in: [typing.ts:129](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L129)

Enum representing failure codes for different error scenarios.

## Enumeration Members

### AuthenticationError

> **AuthenticationError**: `201`

Defined in: [typing.ts:158](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L158)

Indicates an authentication error (e.g., HTTP 401, 403, 407).

***

### ConfigError

> **ConfigError**: `400`

Defined in: [typing.ts:188](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L188)

Indicates an error caused by a misconfigured state.

***

### ConnectionError

> **ConnectionError**: `204`

Defined in: [typing.ts:173](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L173)

Indicates a connection error (e.g., network issues).

***

### EncryptionError

> **EncryptionError**: `301`

Defined in: [typing.ts:183](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L183)

Indicates a error related to the encryption protocol for remote inference.

***

### EngineSpecificError

> **EngineSpecificError**: `300`

Defined in: [typing.ts:178](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L178)

Indicates an engine-specific error.

***

### InvalidArgumentsError

> **InvalidArgumentsError**: `401`

Defined in: [typing.ts:193](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L193)

Indicates that invalid arguments were provided.

***

### InvalidRemoteConfigError

> **InvalidRemoteConfigError**: `402`

Defined in: [typing.ts:198](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L198)

Indicates misconfigured config options for remote inference.

***

### LocalEngineChatError

> **LocalEngineChatError**: `101`

Defined in: [typing.ts:138](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L138)

Indicates a chat error coming from a local engine.

***

### LocalEngineFetchError

> **LocalEngineFetchError**: `102`

Defined in: [typing.ts:143](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L143)

Indicates a fetch error coming from a local engine.

***

### LocalError

> **LocalError**: `100`

Defined in: [typing.ts:133](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L133)

Indicates a local error (e.g., client-side issues).

***

### NoLocalProviderError

> **NoLocalProviderError**: `103`

Defined in: [typing.ts:148](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L148)

Indicates an missing provider for a local model.

***

### NotImplementedError

> **NotImplementedError**: `404`

Defined in: [typing.ts:208](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L208)

Indicates that the requested feature is not implemented.

***

### RemoteError

> **RemoteError**: `200`

Defined in: [typing.ts:153](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L153)

Indicates a remote error (e.g., server-side issues).

***

### TimeoutError

> **TimeoutError**: `203`

Defined in: [typing.ts:168](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L168)

Indicates a timeout error (e.g., HTTP 408, 504).

***

### UnavailableError

> **UnavailableError**: `202`

Defined in: [typing.ts:163](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L163)

Indicates that the service is unavailable (e.g., HTTP 404, 502, 503).

***

### UnknownModelError

> **UnknownModelError**: `403`

Defined in: [typing.ts:203](https://github.com/adap/internal-intelligence/blob/a1d0007cc0e87e7d01df20a73581c407b63dc7ff/intelligence/ts/src/typing.ts#L203)

Indicates an unknown model error (e.g., unavailable or invalid model).
