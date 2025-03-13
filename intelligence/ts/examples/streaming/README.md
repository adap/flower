---
tags: [node, minimal, streaming, typescript]
---

# Response streaming example

You can use `npm` or `pnpm` (or probably `yarn`), but this README shows examples using `pnpm`

## Project setup

You must first download the example with the following command:

```bash
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/intelligence/ts/examples/streaming . && rm -rf _tmp && cd streaming
```

You can then install the project dependencies with:

```bash
pnpm i
```

## Build

If you want to build the project, you can use:

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
