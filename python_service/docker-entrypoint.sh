#!/bin/bash

# Install pydantic if not already installed
pip install pydantic --no-cache-dir

# Run the original command
exec "$@"
