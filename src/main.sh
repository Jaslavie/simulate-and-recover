#!/bin/bash

echo "========================================================"
echo "Running EZ Diffusion Model Simulate-and-Recover Exercise"
echo "========================================================"

cd $(dirname $0)/..
PYTHONPATH=$(pwd) python src/main.py

