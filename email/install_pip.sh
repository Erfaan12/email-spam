#!/bin/sh

# Step 1: Verify Python installation
python --version

# Step 2: Download get-pip.py using curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Step 3: Install pip
python get-pip.py

# Step 4: Verify pip installation
pip --version
