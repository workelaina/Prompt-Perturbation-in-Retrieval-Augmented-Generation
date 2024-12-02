
import csv
import numpy as np
import pickle
import random
# from struct import pack, unpack
# import lzma
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from DGD.help_functions import get_embedding, perturb_sentence, rank_tokens_by_importance, get_initial_ids, user_prompt_generation, check_success, check_MSEloss, get_first_output_token, get_neighbor_ids, check_success, get_relevant_documents
from DGD.DGD import get_optimized_prefix_embedding

device = torch.device("cuda")

# load model and tokenizer
tok = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral', cache_dir="./SFR_Mistral/", torch_dtype=torch.float16)
model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral', cache_dir="./SFR_Mistral/", torch_dtype=torch.float16).to(device)

# no special token
# tok = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral', cache_dir="/SFR_Mistral/", torch_dtype=torch.float16, add_special_tokens=False) 

for param in model.parameters():
    param.requires_grad = False

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


tconst_title = {}
count = 0
with open("./datasets/title.basics.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        if count > 0:
            
            tconst_title[row[0]] = row[2]
            
        count += 1

device = torch.device("cuda")

# %%
prompts = []
name_tt = {}
nconst_name = []
nconst_profession = []
nconst_embedding = []

with open("./datasets/name.basics.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    counter = 0
    
    for row in rd:
        if counter > 1000:
            break
        if counter > 0:            
            prompt = row[1] +" was born in "+row[2]+", "+"and died in "+row[3]+". He/She's primary professions are "+', '.join(map(str, row[4].split(",")))+"."
            tts = row[5].split(",")
            prompt += " He/She is known for movies:"
            for t in tts:
                if t == tts[-1]:
                    prompt += " '" + tconst_title[t]+"'."
                else:
                    prompt += " '" + tconst_title[t]+"',"

            max_length = 526
            input_texts = prompt
            batch_dict = tok(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**batch_dict)
                embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).to("cpu")
        
            nconst_embedding.append(embedding[0])
            prompts.append(prompt)

        counter += 1
        

# SFR mistral
def get_relevant_documents(nconst_embeddings, query, topK, passages, model, tok):
    batch_dict = tok(query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state.to("cpu"), batch_dict['attention_mask'].to(torch.device('cpu'))).to("cpu")
    
    query_embedding = np.array(F.normalize(embeddings, p=2, dim=1)[0])
    docs_embeddings = np.array(F.normalize(nconst_embeddings, p=2, dim=1))

    # get cos similarity
    logits = np.sum(query_embedding[None, :] * docs_embeddings, axis=-1)
    top_k = logits.argsort()[-topK:][::-1]
    print("top_k: ", top_k)
    top_k_docs = [passages[i] for i in top_k]

    return top_k_docs, top_k

def check_success_anotherrr(target_passage_id, top_k):
    if target_passage_id in top_k:
        if top_k[0] == target_passage_id:
            return 2
        return 1
    else:
        return 0

# Set Device
model.eval()

# Set dataset
dataset = "imdb"

iterations = 20
initial_firstk = 5

results_store_dct = {}

for i in range(0, len(prompts)):
    print("----------> prompt index: ", i)
    if i not in results_store_dct:
        results_store_dct[i] = {}

    user_query_id = i
    target_passage_id = i
    
    prompt_text = prompts[i]
    target_passage = prompts[i]

    object = ""
    query_mode = "for_objects"
    user_query = user_prompt_generation(prompt_text, object, dataset, query_mode)

    print("----------> user_query: ", user_query)    
    list_length = 1
    initial_ids = [random.randint(0, 32000) for _ in range(list_length)]
    intital_prefix = tok.decode(initial_ids)

    initial_ids = []

    print("----------> intital_prefix: ", intital_prefix)

    topk_result = 10

    retrieved_docs, top_k = get_relevant_documents(torch.stack(nconst_embedding), user_query, topk_result, prompts, model, tok)

    is_success = check_success_anotherrr(target_passage_id, top_k)
    if is_success > 0:
        print("----------> initial prefix success")

    token_ids = initial_ids
    optimized_prefix = intital_prefix
    results_store_dct[i]["user_query_id"] = i
    results_store_dct[i]["target_passage_id"] = len(prompts) - i - 1
    results_store_dct[i]["prompt_text"] = prompt_text
    results_store_dct[i]["target_passage"] = target_passage
    results_store_dct[i]["token_ids"] = token_ids
    results_store_dct[i]["optimized_prefix"] = optimized_prefix
    results_store_dct[i]["is_success"] = is_success

    print("----------> optimized_prefix: ", optimized_prefix)
    print("----------> user_query_text: ", prompt_text)
    print("----------> target_passage: ", target_passage)

    print("----------> is_success: ", is_success)
    print("---------------> End <--------------")
    print("\n")


# %%
layer = -1
counter_success = 0
counter = 0
total = 0
success_list = []
for json_str in [results_store_dct]:
    for index in json_str:
        if 'is_success' not in json_str[index].keys():
            continue
        if json_str[index]['is_success'] > 0:
            success_list.append(index)
            counter += 1
            if json_str[index]['is_success'] > 1:
                counter_success += 1
            prompt_text = json_str[index]["prompt_text"]
            token_ids = json_str[index]["token_ids"]
            
        total += 1
counter / total, counter_success / total

# # Train adversial prefix
passages_embeddings = F.normalize(torch.stack(nconst_embedding), p=2, dim=1)


# Set Device
device = torch.device("cuda")
model.eval()
MODEL_NAME = "mistralai"

# Set dataset
dataset = "imdb"
query_mode = "for_objects"

iterations = 500
initial_firstk = 10

results_store_dct = {}

for i in range(208, len(prompts)):
    print("----------> prompt index: ", i)
    if i not in results_store_dct:
        results_store_dct[i] = {}

    user_prompt_id = i
    prompt_text = prompts[i]
    object = ""
    user_prompt = user_prompt_generation(prompt_text, object, dataset, query_mode)

    print("----------> user_prompt: ", user_prompt)

    search_range = random.randint(20, 1000)
    target_prompt_id = get_neighbor_ids(user_prompt_id, user_prompt, search_range, model, tok, nconst_embedding)

    print("----------> user_prompt_id: ", user_prompt_id)
    print("----------> prompt_text: ", prompt_text)
    print("----------> target_prompt_id: ", target_prompt_id)
    
    target_text = prompts[target_prompt_id]

    print("----------> target_text: ", target_text)

    initial_firstk = 10
    important_tokens = rank_tokens_by_importance(target_text, model, tok)
    initial_ids = get_initial_ids(important_tokens, target_text, initial_firstk, MODEL_NAME, model, tok)
    intital_prefix = tok.decode(initial_ids.tolist(), skip_special_tokens=True)
    
    print("----------> intital_prefix: ", intital_prefix)

    topk_result = 10
    top_k = get_relevant_documents(torch.stack(nconst_embedding), intital_prefix+user_prompt, topk_result, model, tok)
    is_success = check_success(target_prompt_id, user_prompt_id, top_k)
    
    if is_success > 1:
        token_ids = initial_ids.tolist()
        optimized_prefix = intital_prefix
        print("----------> initial prefix success")
    else:
        print("----------> optimized prompt index: ", i)
        
        thredhold = 1
        token_ids_tensor, optimized_prefix, loss_list = get_optimized_prefix_embedding(MODEL_NAME, passages_embeddings[user_prompt_id], passages_embeddings[target_prompt_id], user_prompt, prompt_text, target_text, initial_ids, iterations, user_prompt_id, target_prompt_id, topk_result, model, tok, nconst_embedding, thredhold)
        token_ids = token_ids_tensor[0].tolist()
        top_k = get_relevant_documents(torch.stack(nconst_embedding), optimized_prefix+user_prompt, topk_result, model, tok)
        is_success = check_success(target_prompt_id, user_prompt_id, top_k)
        
    results_store_dct[i]["user_prompt_id"] = i
    results_store_dct[i]["user_prompt"] = user_prompt
    results_store_dct[i]["target_prompt_id"] = target_prompt_id
    results_store_dct[i]["prompt_text"] = prompt_text
    results_store_dct[i]["target_text"] = target_text
    results_store_dct[i]["token_ids"] = token_ids
    results_store_dct[i]["optimized_prefix"] = optimized_prefix
    results_store_dct[i]["is_success"] = is_success
    results_store_dct[i]["search_range"] = search_range
    results_store_dct[i]["topk_result"] = topk_result

    print("----------> optimized_prefix: ", optimized_prefix)
    print("----------> user_prompt_text: ", prompt_text)
    print("----------> target_text: ", target_text)

    print("----------> is_success: ", is_success)
    print("---------------> End <--------------")
    print("\n")

    with open('./results/SFR_Mistral_IMDB.pkl', 'wb') as file:
        pickle.dump(results_store_dct, file)
