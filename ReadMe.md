# EL2320 Applied Estimation - Project 

This repository contains the code and information regarding the Applied Estimation project. As the project just started, updates are expected. 

## Setup
All code was developed and tested on Windows 10 with Python 3.7.

To run the current [code](GTPreparation), we recommend to setup a virtual environment: 

```bash
python3 -m venv env                     # Create virtual environment
source env/bin/activate                 # Activate virtual environment
pip install -r requirements.txt         # Install dependencies
# Work for a while
deactivate                              # Deactivate virtual environment
```

## Create videos from GT 

In order to create videos out of the given frames just run the following command:

```
python GT_Preparation/RunPrep.py
```
 
If you want to visualize bounding boxes, please ensure that the variable `annotate` is set to `True`. Also, make sure that you have specified the input and output paths correctly.