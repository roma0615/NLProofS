from common import *
import argparse
import torch
import os
import sys
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from abc import ABC, abstractmethod
import json
import numpy as np

class CustomRetriever(ABC):
    @abstractmethod
    def top_k(self, example: Example) -> str:
        pass

class DragonRetriever(CustomRetriever):
    def __init__(self, path: str):
        assert path is not None
        objs = []
        print(f"Initializing DragonRetriever (will read from file)")
        with open(path, "r") as f: # jsonl
            for l in f.readlines():
                objs.append(json.loads(l))
        # create dictionary
        self.hyp_to_context = { x["hypothesis"]: x["context"] for x in objs }

    def top_k(self, example: Example) -> str:
        hyp = example["hypothesis"]
        context = self.hyp_to_context[hyp]
        # number them
        context = [f"sent{i+1}: {x}" for i, x in enumerate(context)]
        # combine into one string
        facts = " ".join(context)
        # print("DRAGON FACTS:", facts)
        return facts

class ContrieverRetriever(CustomRetriever):
    def __init__(self, path: str):
        assert path is not None
        # same code as dragon actually oops
        objs = []
        print(f"Initializing ContrieverRetriever (will read from file)")
        with open(path, "r") as f: # jsonl
            for l in f.readlines():
                objs.append(json.loads(l))
        # create dictionary
        self.hyp_to_context = { x["hypothesis"]: x["context"] for x in objs }

    def top_k(self, example: Example) -> str:
        hyp = example["hypothesis"]
        context = self.hyp_to_context[hyp]
        # number them
        context = [f"sent{i+1}: {x}" for i, x in enumerate(context)]
        # combine into one string
        facts = " ".join(context)
        return facts

# Import SimCSE models
class SimCSERetriever(CustomRetriever):

    def __init__(self, path_corpus: str = None, embeddings_path: str = None, path: str = None, k: int = 25):
        assert ((path_corpus is not None and embeddings_path is not None) and path is None) \
            or ((path_corpus is None and embeddings_path is None) and path is not None), \
            "Must either provide path_corpus and embeddings_path, or path. But not both"
        self.using_hardcoded = path is not None
        self.k = k

        if self.using_hardcoded:
            objs = []
            print(f"Initializing SimCSE retriever (will read from file)")
            with open(path, "r") as f: # jsonl
                for l in f.readlines():
                    objs.append(json.loads(l))
            # create dictionary
            self.hyp_to_context = { x["hypothesis"]: x["context"] for x in objs }
        else:
            print(f"Initializing SimCSE retriever, k={k}")
            self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", use_fast=True) # use_fast = True, does this do anything.
            self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
            self.corpus = self.load_corpus(path_corpus)
            self.corpus_list = list(self.corpus)
            self.embeddings = torch.load(embeddings_path)

    def top_k(self, example: Example) -> str: # return a context string to be passed into extract_context
        
        if self.using_hardcoded:
            hyp = example["hypothesis"]
            context = self.hyp_to_context[hyp]
            # number them
            context = [f"sent{i+1}: {x}" for i, x in enumerate(context)]
            # combine into one string
            facts = " ".join(context)
            return facts
        
        # if not hardcoded... : 
        # example is as found in the jsonl files
        # has context, question, answer, hypothesis, proof (empty for testing), meta obj
        # we only need to edit context given the hypothesis, and return a context string
        # print(f"RETRIEVING TOP K={self.k} FROM GIVEN EXAMPLE HYPOTHESIS {example['hypothesis']}")

        # Tokenize input text
        ex_hypothesis = example["hypothesis"]
        hyp = self.tokenizer([ex_hypothesis], padding=True, truncation=True, return_tensors="pt")
        # Get the embeddings
        with torch.no_grad():
            hyp_emb = self.model(**hyp, output_hidden_states=True, return_dict=True).pooler_output

        # determine the k closest sentences to hypothesis
        scored = {}
        for i, emb in enumerate(self.embeddings):
            scored[i] = 1 - cosine(hyp_emb[0], emb)
        
        scored = sorted(scored.items(), key=lambda x:-x[1]) # [(10, 0.9873), (47, blah), ..., (2, 0.0003)]
        facts = {f"sent{i+1}": self.corpus_list[pair[0]] for i, pair in enumerate(scored[:self.k])}
        # print("FACTS", facts)
        facts = " ".join([f"{k}: {facts[k]}" for k in facts]) # stringify
        # print("Hypothesis:", ex_hypothesis)
        # print(self.k, "retrieved supporting facts:", facts)

        return facts


    def load_corpus(self, path: str):
        with open(path, "r") as f:
            content = f.read()
            corpus = json.loads(content).values()
        return corpus


if __name__ == "__main__":
    retriever = SimCSERetriever(
        "../data/entailment_trees_emnlp2021_data_v3/supporting_data/worldtree_corpus_sentences_extended.json",
        "../data/entailment_trees_emnlp2021_data_v3/supporting_data/worldtree_corpus_simcse_embeddings.pt"
    )
    retriever.top_k({"hypothesis": "northern hemisphere will have the most sunlight in the summer"})