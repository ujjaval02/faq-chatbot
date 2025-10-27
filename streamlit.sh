#!/bin/bash
cd /home/site/wwwroot
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true