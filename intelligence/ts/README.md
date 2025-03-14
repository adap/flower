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
import { FlowerIntelligence } from '@flwr/flwr';

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

You can also quickly try out the library with the [`examples/hello-world-ts`](https://github.com/adap/flower/tree/main/intelligence/ts/examples/hello-world-ts) example (which is a minimal TypeScript project):

```sh
git clone --depth=1 https://github.com/adap/flower.git _tmp && \
mv _tmp/intelligence/ts/examples/hello-world-ts . && \
rm -rf _tmp && \
cd hello-world-ts

pnpm start
```

You can also use `pnpm demo:js` to run the equivalent JavaScript project example ([`examples/hello-world-js`](https://github.com/adap/flower/tree/main/intelligence/ts/examples/hello-world-js)).
