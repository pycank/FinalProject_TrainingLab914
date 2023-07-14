#!/bin/bash

pip install -q kaggle
mkdir -p ~/.kaggle

echo '{"username":"mrhakk","key":"98472b67d59e9c0e3e8d5fa0167bb85c"}' > ~/.kaggle/kaggle.json

ls -lha ~/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
