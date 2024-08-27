import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from angle_emb import AnglE

BASE = "/home/gridsan/cliu/"
BASEDIR = f"{BASE}probing/"


def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['data']


def compute_embeddings(prompts, model,  batch_size=1):
    print("computing embeddings for ", len(prompts), " prompts")
    all_embeddings = []
    for start_index in range(0, len(prompts), batch_size):
        print("processing batch starting at index: ", start_index)
        batch_texts = prompts[start_index:start_index + batch_size]
        try:
            batch_embeddings = model.encode(batch_texts, convert_to_tensor=False)
        except Exception as e:
            print(f"Error encoding batch starting at index {start_index}: {e}")
            continue
        all_embeddings.extend(batch_embeddings)
    res = np.array(all_embeddings)
    print("res shape: ", res.shape)
    return res


def encode_prompts(prompts, model):
    encoded_prompts = []
    for prompt in prompts:
        if prompt is not None:
            encoded = model.encode(prompt)
            # print("encoded: ", encoded)
            encoded_prompts.append(encoded)
    print("length of encoded prompts: ", len(encoded_prompts))
    encoded_array = np.array(encoded_prompts, dtype=object)
    print("encoded_array shape: ", encoded_array.shape)
    return encoded_array


def process_and_save_embeddings(file_path, model, model_name, embedding_suffix, embedding_function):
    records = load_json_data(file_path)
    prompts = [rec['text'] for rec in records if 'text' in rec]

    embeddings_file = os.path.join(BASEDIR,
                                   f"{os.path.splitext(os.path.basename(file_path))[0]}-{embedding_suffix}.npy")
    if not os.path.exists(embeddings_file):
        encoded_prompts = embedding_function(prompts, model)
        print(f"Embeddings computed for {model_name}, saving to {embeddings_file}, shape: {encoded_prompts.shape}")
        np.save(embeddings_file, encoded_prompts)
        print(f"Embeddings saved for {model_name} in {embeddings_file}")
    else:
        print(f"Embeddings file already exists for {model_name}, file: {embeddings_file}")


# List of files to process
files = ["rag-dataset-12000.json", "rag-mini-wikipedia.json",
         "rag_instruct_benchmark_tester.json","mirage-eval-rag-output.json", "rag-mini-bioasq.json"]


# Process each file with UAE model
uae_model = AnglE.from_pretrained(BASE + "hf/UAE-Large-V1", pooling_strategy='cls').cuda()
for file_name in files:
    file_path = os.path.join(BASEDIR, file_name)
    process_and_save_embeddings(file_path, uae_model, "UAE", "uae-embed", encode_prompts)
del uae_model  # Free up memory by deleting the model

# Process each file with SFR model
sfr_model = SentenceTransformer(BASE + "hf/SFR-Embedding-Mistral", device='cuda')
for file_name in files:
    file_path = os.path.join(BASEDIR, file_name)
    process_and_save_embeddings(file_path, sfr_model, "SFR", "sfr-embed", compute_embeddings)
del sfr_model  # Free up memory by deleting the model

