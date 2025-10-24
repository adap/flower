find . -name "*.json" -o -name "*.js" > .files_to_watch
find ./src -name "*.ts" >> .files_to_watch
find ./public -name "*.json" -o -name "*.css" -o -name "*.html" >> .files_to_watch

# Monitor changes to FlowerIntelligence lib
find ../../../dist -name "*.js" -o -name "*.ts" -o -name "*.json" >> .files_to_watch

cat .files_to_watch | entr -c pnpm run ibuild
rm .files_to_watch
