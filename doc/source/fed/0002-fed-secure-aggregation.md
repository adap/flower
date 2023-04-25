---
fed-number: 0002
title: secure aggregation
authors: ["@FANTOME-PAN"]
creation-date: 2023-04-25
last-updated: 2023-04-26
status: provisional
---

# FED Template

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
  - [Data types for SA](#data-types-for-sa)
  - [Server-side components](#server-side-components)
  - [Client-side components](#client-side-components)

## Summary

[//]: # ([TODO - sentence 1: summary of the problem])
The current Flower framework does not have built-in modules for Secure Aggregation (SA).
However, flower users may want to use SA in their FL solutions or
implement their own SA protocols easily.

[//]: # ([TODO - sentence 2: summary of the solution])
Based on the previous SA implementation, I intend to build the SA 
for flower on the Driver API.

[//]: # (## Motivation)

[//]: # ()
[//]: # ([TODO])

[//]: # ()
[//]: # (### Goals)

[//]: # ()
[//]: # ([TODO])

[//]: # ()
[//]: # (### Non-Goals)

[//]: # ()
[//]: # ([TODO])

## Proposal

### Data types for SA

Judging from the SecAgg protocol, the SecAgg+ protocol, the LightSecAgg protocol,
and the FastSecAgg protocol, the following fields can better facilitate
SA implementations.

1. bytes, List of bytes

    SA protocols often use encryption and send ciphertext in bytes.
Besides, cryptography-related information, such as public keys, are normally stored as bytes.
Sharing these info will require transmitting bytes.

    Currently, both FitIns and FitRes contain one dictionary field,
mapping strings to scalars (including bytes).
    Though it is possible to store lists of bytes in the dictionary using tricks,
it can be easier to implement SA if TaskIns and TaskRes have fields supporting bytes and lists of bytes

2. arrays

    In many protocols, the server and the clients need to send 
additional but necessary information to complete SA.
These info are usually single or multiple lists of integers or floats.
We now need to store them in the parameters field.


Considering all above, if possible, I would suggest adding a more general dictionary field,
i.e., Dict[str, Union[Dict[str, LScalar], LScalar]],
where LScalar = Union[Scalar, List[Scalar]]

Alternatively, we can have multiple dictionary fields in addition to the config/metrics dictionary, including:

1. Dict[str, Union[np.ndarray, List[np.ndarray]]]
2. Dict[str, Union[bytes, List[bytes]]]

### Server-side components

The server actively coordinates the SA protocols.
Its responsibilities include:
1. help broadcast SA configs for initialisation.
2. forward messages from one client to another.
3. gathering information from clients to obtain aggregate output.

In short, other then serving as a relay, the server is a controller and decryptor.
It controls the workflow. Since SA protocols are rather different from each other,
we may want to allow customising the workflow, i.e., allowing users to define 
arbitrary rounds of communication in a single FL fit round.

### Client-side components

The key responsibilities of a client are:
1. generate (cryptography-related) information
2. sharing information via the server
3. encrypt its output
4. help the server decrypt the aggregate output

In summary, a client is an encryptor. It requires additional information from 
other encryptors for initialisation and also provides other encryptors with its information.
Then, it can independently encrypt its output. 
In the end of the fit round, it provides the server with necessary information that allows
and only allows the server to decrypt aggregate output, learning nothing of individual outputs.


[//]: # (## Drawbacks)

[//]: # ()
[//]: # ([TODO])

[//]: # ()
[//]: # (## Alternatives Considered)

[//]: # ()
[//]: # (### [Alternative 1])

[//]: # ()
[//]: # ([TODO])

[//]: # ()
[//]: # (### [Alternative 2])

[//]: # ()
[//]: # ([TODO])

[//]: # ()
[//]: # (## Appendix)
