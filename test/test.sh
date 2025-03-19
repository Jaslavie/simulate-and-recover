#!/bin/bash

echo "============================="
echo "Running EZ Diffusion Tests"
echo "============================="

python -m unittest test.test_diffusion_model

echo "Tests complete"