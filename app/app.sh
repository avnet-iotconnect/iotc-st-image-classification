#!/bin/bash

set -e

pushd ~/app/ >/dev/null
source ~/.venv-staiicdemo/bin/activate
python3 app.py "$@"

popd