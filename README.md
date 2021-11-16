# HGL

## Environment
Python =3.6.9, others in `requirements.txt`, you can run
```bash
pip install -r requirements.txt
```

## Config
Please modify the `config.ini` file in configs dir.
(Note that `GPU memory: at least 24 GB`)
```ini
[DEFAULT]
pretrained_model = /share/model/bert/cased_L-24_H-1024_A-16 # should be replaced with yours
model_name = hyper
model_path = model/hyper_org
lr = 3e-6
alpha = 1.0
beta = 0.05
type = ORG
train_dataset = ./data/train.data
batch_size = 150
T = 1.0
hidden_size = 200
cuda_device = 7
num_epochs = 50
seed = 0
do_train = yes
do_eval = yes
[token_indexers]
use_starting_offsets = yes
do_lowercase = no
```
## Run
```bash
python train.py --config-file ./configs/config.ini
```
The result will be saved in model dir.