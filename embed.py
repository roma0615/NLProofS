import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import json
import sys
from common import normalize, extract_context


def preproc_dragon_embeddings(hf_tokenizer: str, hf_cmodel: str, tensor_save_file: str):
    # Import models - the package will take care of downloading the models automatically
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    cmodel = AutoModel.from_pretrained(hf_cmodel) # use the *context* version of the Dragon model (context/query option)
    print("Models loaded")
    
    corpus = load_corpus_sentences("data/entailment_trees_emnlp2021_data_v3/supporting_data/worldtree_corpus_sentences_extended.json")
    print("Corpus loaded")
    ctx_inputs = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")
    
    # Compute token embeddings
    with torch.no_grad():
        ctx_emb = cmodel(**ctx_inputs).last_hidden_state[:, 0, :]
        
    # print(type(ctx_emb), len(ctx_emb))
    if tensor_save_file is not None:
        torch.save(ctx_emb, tensor_save_file)

    return 

def preproc_contr_embeddings(hf_tokenizer: str, hf_model: str, hyp: str, tensor_save_file: str):
    # Import models - the package will take care of downloading the models automatically
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    model = AutoModel.from_pretrained(hf_model)
    print("Models loaded")
    corpus = load_corpus_sentences("data/entailment_trees_emnlp2021_data_v3/supporting_data/worldtree_corpus_sentences_extended.json")
    inputs = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")
    hyp_inp = tokenizer([hyp], padding=True, truncation=True, return_tensors="pt")

    # Compute token embeddings
    with torch.no_grad():
        hyp_emb_out = model(**hyp_inp)
        emb_out = model(**inputs)
    # Mean pooling
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    embeddings = mean_pooling(emb_out[0], inputs['attention_mask'])
    hyp_emb = mean_pooling(hyp_emb_out[0], hyp_inp['attention_mask'])
    
    # SAVE EMBEDDINGS (as tensor)
    if tensor_save_file is not None:
        torch.save(embeddings, tensor_save_file)

    #  As an example: determine the top_k closest sentences to example hypothesis  
    top_k = 25
    scored = {}
    for i, emb in enumerate(embeddings):
        scored[i] = 1-cosine(hyp_emb[0], emb)
    scored = sorted(scored.items(), key=lambda x:-x[1])
    facts = {corpus[i[0]]: i[1] for i in scored[:top_k]}
    print("Hypothesis:", hyp)
    print(top_k, "retrieved supporting facts:", facts)

def preproc_simcse_embeddings(hf_tokenizer: str, hf_model: str, hyp: str, tensor_save_file: str):
    # Import models - the package will take care of downloading the models automatically
    print("Loading HF tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    model = AutoModel.from_pretrained(hf_model)
    ex_hypothesis = hyp
    print("Loading corpus sentences")
    corpus = load_corpus_sentences("data/entailment_trees_emnlp2021_data_v3/supporting_data/worldtree_corpus_sentences_extended.json")
    inputs = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")
    hyp = tokenizer([ex_hypothesis], padding=True, truncation=True, return_tensors="pt")    

    # Compute the embeddings
    print("Computing embeddings")
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        hyp_emb = model(**hyp, output_hidden_states=True, return_dict=True).pooler_output
    # Save embeddings (as tensor) to load in during retrieval
    if tensor_save_file is not None:
        torch.save(embeddings, tensor_save_file)

    # As an example: determine the top_k closest sentences to example hypothesis 
    top_k = 25
    scored = {}
    for i, emb in enumerate(embeddings):
        scored[i] = 1-cosine(hyp_emb[0], emb)
    scored = sorted(scored.items(), key=lambda x:-x[1])
    facts = {corpus[i[0]]: i[1] for i in scored[:top_k]}
    print("Hypothesis:", ex_hypothesis)
    print(top_k, "retrieved supporting facts", facts)


def load_corpus_sentences(path: str):
    corpus = None
    for line in open(path):
        ex = json.loads(line)
        corpus = ex.values()
    return list(corpus)


def main():
    if sys.argv[1] == "simcse":
        tensor_save = "wtcorpus_simcse_embeddings_full.pt"
        hf_model = "princeton-nlp/sup-simcse-bert-base-uncased"
        hf_tokenizer = "princeton-nlp/sup-simcse-bert-base-uncased"
        ex_hyp = "northern hemisphere will have the most sunlight in summer"
        preproc_simcse_embeddings(hf_tokenizer, hf_model, ex_hyp, tensor_save)

    if sys.argv[1] == "contriever":
        contriever_tensor_save = "wtcorpus_contriever_embeddings_full.pt"
        hf_model = 'facebook/contriever'
        hf_tokenizer = 'facebook/contriever'
        ex_hyp = "northern hemisphere will have the most sunlight in summer"
        preproc_contr_embeddings(hf_tokenizer=hf_tokenizer, hf_model=hf_model, hyp=ex_hyp, tensor_save_file=contriever_tensor_save)

    elif sys.argv[1] == "dragon":
        dragon_tensor_save = "wtcorpus_dragon_embeddings_full.pt"
        hf_model = 'facebook/dragon-roberta-context-encoder'
        hf_tokenizer = 'facebook/dragon-roberta-query-encoder'
        preproc_dragon_embeddings(hf_tokenizer, hf_model, dragon_tensor_save)


if __name__ == "__main__":
    main()