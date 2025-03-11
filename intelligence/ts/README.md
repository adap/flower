# Flower Intelligence

You can use `npm` or `pnpm` (or probably `yarn`), but this `README.md` shows examples using `pnpm`

## Install

Install dependencies with:

```sh
pnpm i
```

## Guide

### Instantiation

The `FlowerIntelligence` package uses the singleton pattern. It can be instanciated with:

```typescript
const fi = FlowerIntelligence.instance;
```

### Simple chat

The simplest way to use `FlowerIntelligence` is to pass a simple string to the `chat` method:

```typescript
const response = await fi.chat('How are you?');
```

Or, a list of `Message`:

```typescript
const response = await fi.chat({
  messages: [
    { role: 'system', content: 'You are a helpful assistant' },
    { role: 'user', content: 'How are you?' },
  ],
});
```

The `response` that we get from the `chat` method is either a `Failure` (containing 2 attributes, `code`, a number, and `description`, a string), or
a `ChatResponseResult`, containing `message`, the `Message` object returned by the assistant (of the form
`{role: string, content: string, toolCalls?: ToolCall[]}`).

Note, we can use the `ok` attribute to differentiate between `ChatResponse` and `Failure`.

```typescript
if (!response.ok) {
  console.error(response.failure.code);
} else {
  console.log(response.message.content);
}
```

### Complete code

```javascript
import { FlowerIntelligence } from 'https://unpkg.com/@flwr/flwr';

const fi = FlowerIntelligence.instance;

async function main() {
  const response = await fi.chat({
    messages: [
      { role: 'system', content: 'You are a helpful assistant' },
      { role: 'user', content: 'How are you?' },
    ],
  });

  if (!response.ok) {
    console.error(`${response.failure.code}: ${response.failure.description}`);
  } else {
    console.log(response.message.content);
  }
}

await main().then().catch();
```

## Demo

You can quickly try out the library with the `examples/hello-world` example:

```sh
pnpm demo
```

This script will build the library and run the example demo. You can modify the
prompt used inside `examples/hello-world/index.mjs`.

You can also use `pnpm demo:js-proj` and `pnpm demo:ts-proj` to respectively
run a simple JavaScript project example and a simple TypeScript project example.
Those projects can be found respectively in `examples/simple-js-project` and
`examples/simple-ts-project`. Note that, contrary to `examples/hello-world`,
those project are valid `pnpm`/`npm` projects.

## Build

Compile the library with:

```sh
pnpm build
```

This will compile the TypeScript files to JS, run `vite build`, and compile types.

If you only want to compile types, you can use:

```sh
pnpm build:types
```

## Test

You can run unit tests with:

```sh
pnpm test
```

## Format

You can format the code using:

```sh
pnpm format
```

And check for proper formatting with:

```sh
pnpm format:check
```
