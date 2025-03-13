---
tags: [web, vision, fds, wandb]
dataset: [Fashion-MNIST]
framework: [torch, torchvision]
---

# Flower Intelligence Flower Extension Example

## Dev Notes

- The `watch` script relies on [entr](https://github.com/eradman/entr).
- You can use `npm` or `pnpm` (or probably `yarn`), but this README shows examples using `pnpm`

Install dependencies: `pnpm i`

In order to use remote handoff, copy `.env.example` into `.env` and update the API keys and IDs inside it.

Build with:

```sh
pnpm build
```

Or rebuild when files change:

```sh
# Make sure to install this first: https://github.com/eradman/entr
pnpm watch
```
