---
tags: [web, browser, extension, chat, typescript]
---

# Browser Extension Example

You can use `npm` or `pnpm` (or probably `yarn`), but this README shows examples using `pnpm`

## Project setup

You must first download the example with the following command:

```bash
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/intelligence/ts/examples/encrypted . && rm -rf _tmp && cd encrypted
```

You can then install the project dependencies with:

```bash
pnpm i
```

> [!NOTE]
> The `watch` script relies on [entr](https://github.com/eradman/entr).

> [!WARNING]
>In order to use remote handoff, copy `.env.example` into `.env` and update the API key inside it (if you don't have a valid API key, you can register [here](https://flower.ai/intelligence/)).
>
>```typescript
>fi.apiKey = 'REPLACE_HERE';
>```

## Build

Then, you need to build the project:

```bash
pnpm build
```

Or rebuild when files change:

```sh
# Make sure to install this first: https://github.com/eradman/entr
pnpm watch
```

## Run

Once you have built the project, you should find the web-extension content that 
can be imported into a browser in `dist/`.
