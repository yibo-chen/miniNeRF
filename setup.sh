#!/bin/bash

ENV_NAME=".venv"

python3 -m venv $ENV_NAME

source $ENV_NAME/bin/activate

pip3 install -r requirements.txt

echo "Setup is complete!
To activate the virtual environment: 'source $ENV_NAME/bin/activate'
To deactivate the virtual environment, type 'deactivate'."
