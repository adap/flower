---
fed-number: 0002
title: Experimental REST Transport Layer
authors: ["@danieljanes"]
creation-data: 2022-10-26
last-updated: 2022-10-26
status: provisional
---

# Experimental REST Transport Layer

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
- [Design Details](#design-details)
- [Drawbacks](#drawbacks)
- [Alternatives Considered](#alternatives-considered)

## Summary

A new REST-based transport layer would enable Flower clients to connect to a Flower server using ...

## Motivation

Flower's gRPC-based connectivity stack has been successfully adopted in both research and production scenarios. However, there are some environments in which the reliance on bidirectional gRPC streams makes it challenging for users to deploy Flower. This can be either due to the usage of gRPC itself (gRPC is not supported on some platforms) or due to the current implementation relying on bidirectional streams (long-standing connections are not yet well-supported in some cloud environments).

The motivation for this FED is, therefore, to make it easier for users to integrate, use, and deploy Flower in environments that have traditionally not been well supported.

### Goals

- *Increase infrastructure compatibility.* default configurations of common cloud infrastructure stacks.
  - *Be compatible with caching infrastructure.*
  - *Support CORS.*
- *Increase client platform compatibility.* Flower clients should be able to run on platforms that are not well-supported by gRPC.
- *Support proxy servers.* Enable client/server connections to go through proxy servers that may or may not alter the request, for example, to conceal the client's IP address from the server.
- *Keep client identity optional.* Clients should be able to remain anonymous, client identity should be an optional component that users of the framework can opt for depending on the needs of their workload.
- *Prevent clients from poisoning the model.*
- *Limit server-side knowledge about individual clients.*

### Non-Goals

- *End-to-end latency.* The existing bidirectional stream between client and server offers stellar end-to-end latency (i.e., from scheduling a task for a client until that client starts executing the task locally). Most workloads do not require such low levels of latency.
- *API stability.* The proposed REST API is not intended to be final.
- *Client authentication.* There should be a forward-compatible path for adding client authentication, but client authentication does not need to be implemented in the first version.
- *Compatibility with `start_server`.*
- *Build a "RESTful" API.* The goal is to have a REST-inspired API, but not to build a fully idiomatic RESTful API.

## Proposal

Introduce a request/response REST transport layer. The REST transport layer would consist of a REST API server (integrated with the Flower server) and a REST client.

## Design Details

### Definition(s)

Task: User-defined specification of what a client has to do, with the expectation that the client returns a result.

### Protocol

General idea:
- Client always initiates conversation
- Server responds in one of two ways:
  - Reconnect at a later time.
  - Execute a task and send the result.
- Server has a (mild) form of control over the conversation by suggesting to the client when it should connect again. This can be used, among other things, to implement pace steering.

Example:
- Client connects to the server: "I'm available"
- Server doesn't have any tasks: "reconnect in 5min"
- Client sleeps for 5min, then connects again
- Server say: here's a set of TASK-TOKEN-PAIRS, please work on them and send me the RESULTS
- Client receives the set of TASK-TOKEN-PAIRS, decides which ones to opt-in to, executes the chosen ones locally, and returns the RESULT
  - Client opt-in provides a possibility for local task rejection, including plausible deniability because it is not certain that a client will work on a task
  - Opt-in criteria are up to the client, examples include connectivity status, charging status, availability of local training/evaluation data, and manual approval (e.g., in a cross-silo setting)
  - Local task execution can happen sequentially or concurrently (or any mix of the two), depending on what the client chooses to do
  - Result submission is also up to the client, the client can submit one result after another or batch multiple results together
- Server says thanks, please reconnect in 10min
- [the process repeats]

### Server-side REST API

#### Task/result calls

Situation: a `Task`/`TaskAssignment` scheduled for `N` anonymous clients

**GET /tasks**
Request body: `None`
Response body: `GetTasksResponse` containing either `Task` + `token` or `Reconnect`

Implementation:
1. Is there any task scheduled? Yes, one for `N` anonymous clients
2. Generate the first `token`, store it with the `Task`/`TaskAssignment` in the `State`
3. Return `Task` + `token` to the client in the `GetTasksResponse`

**POST /results**

Request body: `SubmitResultRequest` containing `Result` + `token`
Response body: `None`

Implementation:
1. Check if the `token` is known
2. Save `Result` if true, else return error

#### Client availability calls

**POST /available**


### Message serialization

TODO:
- Continue to use ProtoBuf for serialization.
- Send/receive serialized ProtoBuf messages.
- `Content-Type` header must be set to `application/protobuf`.
- Clients are expected to send an `Accept` header indicating that they accept `application/protobuf` responses. Clients must check the response `Content-Type` to ensure forward-compatibility.

### Message types

Messages types should stay identical to the existing ones, particularly the ones used in the Driver API:

- `Task` wrapping a `ServerMessage` with its nested message types (`FitIns`, `EvaluateIns`, ...)
- `Result` wrapping a `ClientMessage` with its nested message types (`FitRes`, `EvaluateRes`, ...) 

### Security considerations

**HTTPS-only:** Clients are expected to connect to the server only via HTTPS, never via plain HTTP.

**Poisoning attacks:** [TODO, which mechanisms do we want to support/enforce here?]

### Caching

TODO

### Basic model poisoning prevention

TODO

## Drawbacks

The proposed design comes with a few drawbacks:

- Platforms that do not support gRPC still need a way to serialize/deserialize ProtoBuf messages.
- Request/response will most likely result in increased end-to-end latency.

## Alternatives Considered

### Request/response gRPC

A new transport layer that continues to use gRPC, but replaces bidirectional streaming with a more traditional request/response model, would offer a lot of the benefits of the proposed REST-based request/response approach. It might also outperform the REST-based approach.

We propose to use REST because REST increases compatibility with platforms that lack gRPC support. Many organizations also have substantially more experience operating REST-based services, which it easier for them to adopt and operate a REST-based system.

### JSON serialization

REST APIs often use JSON as their preferred serialization format. We propose to continue to use ProtoBuf for serialization, at least for the initial version of the REST transport layer. Using ProtoBuf comes with several advantages:

- Consistency with the existing gRPC transport layer: no risk of introducing inconsistencied between ProtoBuf messages and JSON messages.
- No duplication: no need to maintain two serialization formats.
- Reduced effort: no need to implement JSON serialization.
- Smaller request/response body size: ProtoBuf messages are substantially smaller compared to JSON messages.

Setting the `Accept` and `Content-Type` headers also provides a forward-compatible path to use other serialization formats (like JSON) in the future, if required.
