#!/bin/bash

echo "============================="
echo "Running EZ Diffusion Tests"
echo "============================="

python -m unittest discover test

echo "Tests complete"