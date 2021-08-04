#!/bin/sh

# Ensure that pip in installed
# python -m ensurepip --default-pip
# pip3 install -r requirements.txt

# Run the machine learning model
python3 ./src/run.py
wait
