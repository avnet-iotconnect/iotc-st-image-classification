#!/bin/bash

set -e

cd "$(dirname "$0")"

CONFIG_BOARD=$(find /usr/local/x-linux-ai -name "config_board_*.sh")
source $CONFIG_BOARD

pushd ~/app/ >/dev/null
source ~/.venv-staiicdemo/bin/activate
python3 app.py "$@"

popd