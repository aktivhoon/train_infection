#add gym_infection path to environment path
#import os,sys
#sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
from config import simple2d_config
from utils import get_n_params
from trainer import PPOTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
run_id = 'ppo'

trainer = PPOTrainer(simple2d_config(), run_id=run_id, device=device)
print('model_weights_num', get_n_params(trainer.model))
trainer.run_training()
trainer.close()
