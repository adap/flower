#!/bin/bash -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"/../

EXIT_CODE=0

while IFS= read -r -d '' file; do
    COPYRIGHT=$(grep -h -r "copyright = " "$file")
    if [ "$COPYRIGHT" != "copyright = f\"{datetime.date.today().year} Flower Labs GmbH\"" ]; then
        echo "::error file=$file::Wrong copyright line. Expected: copyright = f\"{datetime.date.today().year} Flower Labs GmbH\""
        EXIT_CODE=1
    fi
done < <(find . -path "*/doc/source/conf.py" -print0)

exit $EXIT_CODE
