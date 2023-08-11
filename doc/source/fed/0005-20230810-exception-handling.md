---
fed-number: 0005
title: Exception Handling
authors: ["@adap"]
creation-date: 2023-08-10
last-updated: 2023-08-11
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

## Summary

### Problem
The current Flower framework lacks exception handling, and thus any exception in a non-simulated client, even minor ones, can terminate the its entire process. Addressing minor, recoverable exceptions without interupting the process and reporting them to the server is crucial (but not really urgent).

### Solution
On the client side, adopting a fixed list of recoverable exceptions and allow users to extend it by defining their own exception classes inheriting from our base exception class `Recoverable`.
On the server side, we should implement a feature that exposes the exceptions collected from clients to the driver.

## Motivation

A mechanism that handles recoverable exceptions properly can greatly increase the robustness of clients, and consequently FL algorithms running under our framework.
Additionally, it allows users to report reoverable failures to the driver without having to terminate the client or figure out a compromised way to store error messages in normal replied messages.

### Goals

- Identify a list of potentially recoverable exceptions.
- Finalize the list of recoverable exceptions.
- Introduce the base class `Recoverable` for all flower-specific exceptions and user-defined exceptions. It should be placed in `flwr.errors` (and yes, we don't currently have this, so I am open to change).
- Update gPRC messages to facilitate exception reporting.
- Update the server-side code to receive exception messages from clients and redeliver them to drivers.
- Implement the client-side exception handling.

### Non-Goals

- Investigate and handle python-library-specific exceptions, such as errors raised by `tensorflow`, `pytorch`, `numpy`, etc.
- Extend the field of discussion to other exceptions not related to client-side message handlers.

## Proposal

### List of Exceptions
As this point, we should **only** consider exceptions that are related to client-side message handlers. The following exceptions are [built-in exceptions in Python](https://docs.python.org/3/library/exceptions.html#) , but it may be reasonable to include the exceptions raised by flower dependencies in the future.

 >IMO, the most important criterion for including an exception or not is that, after capturing this exception, whether the driver is willing to analyze it and then adjust the future instructions to avoid such an exception.


Exceptions in the list is **potentially** recoverable. The list is not finalized.

- [ ] `TimeoutError`: Raised when a system function timed out at the system level.
- [ ] `MemoryError`: Raised when an operation runs out of memory but the situation may still be rescued (by deleting some objects). **Handling this exception is vital for many heterogeneous FL algorithms that want to personalize tasks for each client based on its budget.**
- [ ] `FileNotFoundError`: Raised when a file or directory is requested but doesn’t exist.
- [ ] `FileExistsError`: Raised when trying to create a file or directory which already exists.
- [ ] `IsADirectoryError`: Raised when a file operation (such as os.remove()) is requested on a directory.
- [ ] `NotADirectoryError`: Raised when a directory operation (such as os.listdir()) is requested on something which is not a directory.
- [ ] `PermissionError`: Raised when trying to open a file in write mode where the file is only available for read mode, or vice versa.
- [ ] `EOFError`: Raised when the input() function hits an end-of-file condition (EOF) without reading any data.
- [ ] `ValueError`: Raised when a function gets an argument of the correct type but improper value.
- [ ] `KeyError`: Raised when a dictionary key is not found.
- [ ] `IndexError`: Raised when a sequence subscript (index) is out of range.
- [ ] `TypeError`: Raised when an operation or function is applied to an object of inappropriate type.
- [ ] `OSError`: General purpose error, related to system or I/O operation failures. Specific instances of this error might be recoverable depending on the context.

<!-- ## Discussion

There are a few places that remain unclear. First, what gRPC message should be used to carry the exception information? Stroing the info in `TaskRes` can be a natural way, but this will require adding additional fields to the `TaskRes`. Specifically, if we want to add  -->