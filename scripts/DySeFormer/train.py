import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from util.tools import load_yaml_config

config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
config = load_yaml_config(config_path)

# basic learning set
itr = config['training']['itr']
seq_len = config['training']['seq_len']
pred_len = config['training']['pred_len']
patience = config['training']['patience']
train_epochs = config['training']['train_epochs']
learning_rate = config['training']['learning_rate']

# backbone model
model_name = 'DySeFormer'

# model setting
n_blocks = config['model']['n_blocks']
e_layers = config['model']['e_layers']
d_model = config['model']['d_model']
n_heads = config['model']['n_heads']
d_ff = config['model']['d_ff']

patch_len = config['model']['patch_len']
stride = config['model']['stride']
padding_patch = config['model']['padding_patch']

dropout = config['model']['dropout']
fc_dropout = config['model']['fc_dropout']
head_dropout = config['model']['head_dropout']

topk = config['model']['topk']
revin = config['model']['revin']

#gpus
use_multi_gpu = config['gpu']['use_multi_gpu']
devices = config['gpu']['devices']

# datasets
root_path = os.path.join(ROOT_DIR, 'datasets')
data = config['dataset']['data']
batch_size = config['dataset']['batch_size']
features = config['dataset']['features']

# time freq
freq = config['model']['freq']
use_timestamp = config['model']['use_timestamp']

os.system(f"python run_simple_Exp.py --itr {itr} \
          --seq_len {seq_len} --pred_len {pred_len} \
          --learning_rate {learning_rate} --patience {patience} --features {features}\
          --data {data} --model {model_name}\
          --root_path {root_path} --batch_size {batch_size} --train_epochs {train_epochs}\
          --n_blocks {n_blocks} --e_layers {e_layers} --d_model {d_model} --n_heads {n_heads} --d_ff {d_ff} \
          --patch_len {patch_len} --stride {stride} --padding_patch {padding_patch} \
          --dropout {dropout} --fc_dropout {fc_dropout} --head_dropout {head_dropout} \
          --topk {topk} --revin {revin} --freq {freq} --use_timestamp {use_timestamp}\
          --use_multi_gpu {use_multi_gpu} --devices {devices}"
          )
