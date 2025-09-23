# DySeFormer
This repository maintains the official implementation of methods and experiments presented in our paper titled "DySeFormer: Dynamic Selective Transformer for Multivariate Time Series Forecasting"

## Preparation for Environment
### 1. Create a conda environment
Our code was tested with Python 3.10.
```
conda create -n DySeMTS python=3.10
```
### 2. Activate the Conda Environment
```
conda activate DySeMTS 
```
### 3.Install the Requirements
```
pip install -r requirements.txt
```
## Scripts
An example:
```
python run_simple_Exp.py --dataset ETTm1 --model DySeFormer --seq_len 336 --pred_len 96 --learning_rate 0.001
```
All script commands are located in scripts/DySeFormer. The hyperparameters can be configured in config.yaml, and you are free to adjust them according to your needs.