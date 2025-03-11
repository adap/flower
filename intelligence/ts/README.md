# Flower Intelligence

Check out the full documentation [here](https://flower.ai/docs/intelligence), and the project website [here](https://flower.ai/intelligence).

You can use `npm` or `pnpm` (or probably `yarn`), but this `README.md` shows examples using `pnpm`

## Install

To install via NPM, run:

```sh
pnpm i @flwr/flwr
```

Alternatively, you can use it in vanilla JS, without any bundler, by using a CDN or static hosting. For example, using ES Modules, you can import the library with:

```html
<script type="module">
    import { FlowerIntelligence } from 'https://cdn.jsdelivr.net/npm/@flwr/flwr';
</script>
```

## Hello, Flower Intelligence!

```javascript
// If installed with NPM
// import { FlowerIntelligence } from '@flwr/flwr';

import { FlowerIntelligence } from 'https://cdn.jsdelivr.net/npm/@flwr/flwr';

const fi = FlowerIntelligence.instance;

async function main() {
  const response = await fi.chat({
    model: 'meta/llama3.2-1b/instruct-fp16',
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

You can also quickly try out the library with the `examples/hello-world` example:

```sh
pnpm demo
```

This script will build the library and run the example demo. You can modify the
prompt used inside `examples/hello-world/index.mjs`.

You can also use `pnpm demo:js-proj` or `pnpm demo:ts-proj` to respectively
run a simple JavaScript project example or a simple TypeScript project example.
Those projects can be found respectively in `examples/simple-js-project` and
`examples/simple-ts-project`. Note that, contrary to `examples/hello-world`,
those project are valid `pnpm`/`npm` projects.
