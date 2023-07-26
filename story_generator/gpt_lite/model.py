from utils import data
import torch
import pickle

### Model Inference ###
# Define data loading parameters.
batch_size = 64 # 64 # 16
context_size = 256 # 256 # 32
max_iterations = 5000
eval_interval = 500 # 500 # 100
learning_rate = 3e-4 # 3e-4 # 1e-3
eval_iters = 200
n_embeddings = 384 # 384 # 64
n_heads = 6 # 6 # 4
n_layers = 6 # 6 # 4
dropout = 0.2 # 0.2 # 0.0
train_test_split = 0.9
# Explore input

# Load GPU.
device = torch.device('cpu')
model_inputs = {}

with open('story_generator/gpt_lite/input_shakespeare.txt', 'r', encoding='utf-8') as f:
    text_shakespeare = f.read()
vocab_size, _, _, _, _, stoi, itos = data(text_shakespeare, train_test_split)
model_inputs['shakespeare'] = [vocab_size, n_embeddings, n_heads, n_layers, context_size, dropout, device, stoi, itos]

with open('story_generator/gpt_lite/input_seuss.txt', 'r', encoding='utf-8') as f:
    text_seuss = f.read()
vocab_size, _, _, _, _, stoi, itos = data(text_seuss, train_test_split)
model_inputs['seuss'] = [vocab_size, n_embeddings, n_heads, n_layers, context_size, dropout, device, stoi, itos]

pickle.dump(model_inputs, open('story_generator/gpt_lite/model_variables.pkl','wb'))