#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

ROOT=`pwd`
INDEX=$ROOT/examples/README.md
INSERT_LINE=6

sed -i.bu '7,$d' $INDEX

cd examples/
for d in */ ; do
  example=${d%/}
  [[ $example = doc ]] || cp $example/README.md $ROOT/examples/doc/source/$example.md 2>&1 >/dev/null
  [[ $example = doc ]] || (echo $INSERT_LINE; echo a; echo $example; echo .; echo wq) | ed $INDEX 2>&1 >/dev/null
done

echo "\`\`\`" >> $INDEX

mv $INDEX $ROOT/examples/doc/source/index.md
