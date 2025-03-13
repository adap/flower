---
tags: [node, minimal, remote, encryption, Flower Confidential Remote Compute, typescript]
---

# Flower Confidential Remote Compute example

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

> [!WARNING]
>In order to run this example, you need to update the following values inside `src/index.ts` with a valid API key (if you don't have one, you can register [here](https://flower.ai/intelligence/)):
>
>```typescript
>fi.apiKey = 'REPLACE_HERE';
>```

## Build

Then, you need to build the project:

```bash
pnpm build
```

## Run

In order to run the example once the project has been built:

```bash
node dist/index.js
```

> [!NOTE]
> You can also use `pnpm start` to perform the installation, build, and run steps at the same time.
