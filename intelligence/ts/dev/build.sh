#!/usr/bin/env bash

# Clean and set up
rm -rf dist/*

tsc

# Build the vue app
vite build

vite build --config vite.config.bundled.ts
cp dist/bundled/flowerintelligence.bundled.es.js dist/
rm -rf dist/bundled

pnpm build:types
