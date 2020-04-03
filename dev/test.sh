#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

isort --skip src/flower/proto --skip src/flower_example --check-only -rc src   && echo "- isort:  done" &&
black --exclude "src\/(flower\/proto|flower_example)" --check src              && echo "- black:  done" &&
mypy src                                                                       && echo "- mypy:   done" &&
pylint --ignore=src/flower/proto src/flower src/flower_tool                    && echo "- pylint: done" &&
pytest src/flower src/flower_tool                                              && echo "- pytest: done" &&
echo "- All Python checks passed"
