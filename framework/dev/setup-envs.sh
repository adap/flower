#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

SHELL_BIN=$(basename "$SHELL")

case "$SHELL_BIN" in
  *"bash"*) CONFIG_FILE="$HOME/.bashrc"
  ;;
  *"zsh"*) CONFIG_FILE="$HOME/.zshrc"
  ;;
  *"ksh"*) CONFIG_FILE="$HOME/.kshrc"
  ;;
  *) echo "Shell config file not found, you might want to add 'export FLWR_TELEMETRY_ENABLED=0' manually to it." && exit 0
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
