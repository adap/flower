# Flower Intelligence - TypeScript

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

And configured with:

```typescript
fi.remoteHandoff = true; // False by default
fi.apiKey = '<API_KEY>'; // Undefined by default
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

### Streaming

In order to stream the response from the model, you can set `stream: true`, and provide a callback to the `chat` mehods options.
The callback should take a `StreamEvent` argument of the form `{ chunk: string }`.

```typescript
const response = await fi.chat({
  messages: [
    { role: 'system', content: 'You are a helpful assistant' },
    { role: 'user', content: 'How are you?' },
  ],
  stream: true,
  onStreamEvent: (event: { chunk: string }) => {
    process.stdout.write(event.chunk);
  },
});
```

Note that once the streaming is done, the full response is returned by the `chat` method:

```typescript
if (!response.ok)) {
  console.error(response.failure.code);
} else {
  console.log(`\n\nComplete reply: ${response.message.content}`);
}
```

### Tool calling

In order to utilize tool calling, you need to pass a `Tool` list to the `tools` option of `chat`.
The format is the one used by OpenAI, and we plan to offer a way to quickly serialize functions into it.

```typescript
const response = await fi.chat({
  messages: [
    {
      role: 'user',
      content: 'Can you draft an email about my football game to my friend Tom?',
    },
  ],
  tools: [
    {
      type: 'function',
      function: {
        name: 'draftEmail',
        description: 'Draft an email for a given receiver',
        parameters: {
          type: 'object',
          properties: {
            receiver: {
              type: 'string',
              description: 'The name of the person the email should be sent to.',
            },
            content: {
              type: 'string',
              description: 'The content of the email to send.',
            },
          },
          required: ['receiver', 'content'],
        },
      },
    },
  ],
});
```

Here is an example of how the resulting `toolCalls` can be processed:

```typescript
function draftEmail({ receiver, content }: { [argName: string]: string }) {
  // Implementation
}

const functionsMap = {
  draftEmail,
};

if (!response.ok) {
  console.error(response.failure.code);
} else {
  if (response.message.toolCalls) {
    const tool = response.message.toolCalls.pop();
    if (tool) {
      functionsMap[tool.function.name as keyof typeof functionsMap](tool.function.arguments);
    }
  }
}
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

Finally, you can try out tool calling with `pnpm demo:tool` and `pnpm demo:tool-ts`.

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
