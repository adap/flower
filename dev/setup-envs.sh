#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

BASH_FILE="$HOME/.bash_profile"

if  ! grep -q "FLWR_TELEMETRY_ENABLED" $BASH_FILE ; then
  echo "Writing FLWR_TELEMETRY_ENABLED=0 into .bash_profile."
  echo "Use new terminal for it to be loaded."

  echo "" >> $BASH_FILE
  echo "# Flower config" >> $BASH_FILE
  echo "export FLWR_TELEMETRY_ENABLED=0" >> $BASH_FILE
fi
