---
tags: [web, chat, minimal, typescript]
---

# Web Chat example

You can use `npm` or `pnpm` (or probably `yarn`), but this README shows examples using `pnpm`

## Project setup

You must first download the example with the following command:

```bash
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/intelligence/ts/examples/web-chat . && rm -rf _tmp && cd web-chat
```

And install the required dependencies:

```bash
pnpm i
```

## Run

In order to run the project, you can use:

```bash
pnpm dev
```

This should display a URL that you can navigate to on your browser to view the web-chat.

```{note}
You can also use `pnpm start` to perform the installation, build, and run steps at the same time.
```
