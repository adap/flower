#!/usr/bin/env bash

# Clean and set up
rm -rf dist/*

tsc

# Build the vue app
vite build

vite build --config vite.config.bundled.ts
cp -a dist/bundled/. dist/
rm -rf dist/bundled

pnpm build:types
