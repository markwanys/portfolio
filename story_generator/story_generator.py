from flask import Blueprint, request, render_template
from story_generator.gpt_lite.utils import encode, decode, set_seed, BigramModel
import html
import torch
import pickle
import json

model_inputs = pickle.load(open('story_generator/gpt_lite/model_variables.pkl', 'rb'))

story_bp = Blueprint("story", __name__, template_folder="templates", static_folder="static", static_url_path='/story_generator/static')

@story_bp.route('/')
def home():
    return render_template('story_generator/home.html')

@story_bp.route('/predict', methods=['POST'])
def predict():
    genre, max_new_tokens, context, seed = (x for x in request.form.values())
    max_new_tokens = int(max_new_tokens)
    set_seed(seed)

    vocab_size, n_embeddings, n_heads, n_layers, context_size, dropout, device, stoi, itos = model_inputs[genre]
    model = BigramModel(vocab_size, n_embeddings, n_heads, n_layers, context_size, dropout, device)
    model.load_state_dict(torch.load(f"story_generator/gpt_lite/save_files/{genre}_state.pt", map_location=device))

    context_enc = torch.tensor(encode(context, stoi), dtype=torch.long, device=device)[None,:]
    # context_enc = torch.zeros((1,1), dtype=torch.long, device=device)
    output = decode(model.generate(context_enc, max_new_tokens=max_new_tokens, context_size=context_size)[0].tolist(), itos)
    output = html.escape(output)
    print(output)
    
    return render_template('story_generator/home.html', prediction_text=json.dumps(output),
                                                        genre=genre,
                                                        max_new_tokens=max_new_tokens,
                                                        context=context,
                                                        seed=seed
                                                        )

