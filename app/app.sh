#!/bin/bash

set -e

if [ -f ~/.aws-env.sh ]; then
  source  ~/.aws-env.sh
fi

pushd ~/app/ >/dev/null
source ~/.venv-staiicdemo/bin/activate
python3 app.py "$@"

popd