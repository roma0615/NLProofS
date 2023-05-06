from common import *
import argparse
import torch
import os
import sys
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

import json

def main():
    parser = argparse.ArgumentParser(description="Preprocess the WorldTree V2 corpus with SimCSE embeddings.")
    parser.add_argument(
        "--src",
        type=str,
        default="data/entailment_trees_emnlp2021_data_v3/supporting_data/worldtree_corpus_sentences_extended.json",
        help="Path to the WorldTree corpus.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/entailment_trees_emnlp2021_data_v3/supporting_data/worldtree_corpus_simcse_embeddings.pt",
        help="Path to save SimCSE embeddings tensor file",
    )
    args = parser.parse_args()
    print(args)

    if os.path.exists(args.dst):
        print(f"Destination file {args.dst} already exists.")
        inp = input("Continue anyway? (y/N)")
        if inp.lower().strip() != "y":
            sys.exit(1)
        print("Overwriting...")
        
    

    # Import SimCSE models
    print("Downloading SimCSE models")
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", use_fast=True) # use_fast = True
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    # Tokenize input texts
    # ex_hypothesis = "northern hemisphere will have the most sunlight in summer"
    print("Loading corpus sentences")
    corpus = load_corpus_sentences(args.src)
    print(f"{len(corpus)} examples loaded")
    # Q: very large (12k) -- should we batch this or something? 

    print("Tokenizing")
    inputs = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt") # batched=True ??
    # will do this step in custom data module
    # hyp = tokenizer([ex_hypothesis], padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    print("Creating embeddings")
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        # hyp_emb = model(**hyp, output_hidden_states=True, return_dict=True).pooler_output


    # SAVE SIMCSE EMBEDDINGS (as tensor)
    print(f"Saving embeddings to {args.dst}")
    torch.save(embeddings, args.dst)

    # # determine the top_k closest sentences to hypothesis 
    # top_k = 25
    # scored = {}
    # for i, emb in enumerate(embeddings):
    #     scored[i] = 1-cosine(hyp_emb[0], emb)
    
    # scored = sorted(scored.items(), key=lambda x:-x[1])
    # facts = {corpus[i[0]]: i[1] for i in scored[:top_k]}
    # print("Hypothesis:", ex_hypothesis)
    # print(top_k, "retrieved supporting facts", facts)


def load_corpus_sentences(path: str):
    corpus = None
    for line in open(path):
        ex = json.loads(line)
        corpus = ex.values()
    return list(corpus)

if __name__ == "__main__":
    main()
