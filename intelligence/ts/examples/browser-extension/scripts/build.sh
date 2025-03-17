# Clean and set up
rm -rf dist/*
mkdir -p dist/assets

# Build `options.js` as a library
vite build --config vite.config.options.js
cp -R dist/options/*.js dist/
rm -rf dist/options

# Build `options.js` as a library
vite build --config vite.config.popup.js
cp -R dist/popup/*.js dist/
rm -rf dist/popup

# Copy contents of `public` dir
cp -R public/* dist/
