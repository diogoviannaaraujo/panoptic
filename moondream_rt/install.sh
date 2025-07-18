#! /bin/bash

if [ -d ".venv" ]; then
  echo "venv exists"
else
  python3 -m venv .venv
fi

./.venv/bin/pip install -r requirements.txt

