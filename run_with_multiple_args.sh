#!/bin/zsh

# List of argument parameters
args=("-m LSTM" "-m Transformer" "-m S4")

# Loop through the argument parameters
for arg in "${args[@]}"
do
  # Run Python script with current argument parameter
  python tandemaus_main.py "$arg"
  
  # Wait for Python script to finish running
  wait
done
