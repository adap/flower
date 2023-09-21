SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for filename in $SCRIPT_DIR/test_*; do python $filename; done