#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

case "$SHELL" in
  "/bin/bash") CONFIG_FILE="$HOME/.bashrc"
  ;;
  "/bin/zsh") CONFIG_FILE="$HOME/.zshrc"
  ;;
  "/bin/ksh") CONFIG_FILE="$HOME/.kshrc"
  ;;
esac

if  ! grep -q "FLWR_TELEMETRY_ENABLED" $CONFIG_FILE ; then
  echo "Writing FLWR_TELEMETRY_ENABLED=0 into $CONFIG_FILE"
  echo "Use new terminal for it to be loaded."

  echo "" >> $CONFIG_FILE
  echo "# Flower config" >> $CONFIG_FILE
  echo "export FLWR_TELEMETRY_ENABLED=0" >> $CONFIG_FILE
else
  echo "Telemetry is already disabled."
fi
