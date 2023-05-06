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

class IterativeRetriever(ABC):
    @abstractmethod
    def top_k(self, sentences: List[str]) -> OrderedDict[str, str]: # dict of uuid and sentence
        pass

class DragonIterativeRetriever(IterativeRetriever):
    def __init__(self, path_corpus: str, embeddings_path: str, k: int = 25):
        assert path_corpus is not None
        assert embeddings_path is not None
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-roberta-query-encoder", use_fast=True) # use_fast = True, does this do anything.
        self.model = AutoModel.from_pretrained("facebook/dragon-roberta-context-encoder")
        self.corpus = self.load_corpus(path_corpus)
        self.corpus_list = list(self.corpus)
        self.embeddings = torch.load(embeddings_path)
        self.k = k

    # TODO copy from score_retrieval.py !!! this is wrong!
    def top_k(self, sentences: List[str]) -> OrderedDict[str, str]:
        # Tokenize input text
        sents = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        # Get the embeddings
        with torch.no_grad():
            sent_embeds = self.model(**sents).last_hidden_state[:, 0, :]

        # determine the k closest sentences to hypothesis
        # cosine similarity 
        # scores[i] will be for a given thing
        rest_of_corpus = 
        scores = np.zeros((len(sentences), len()))
        scored = {}
        for i, emb in enumerate(self.embeddings):
            scored[i] = 1 - cosine(hyp_emb[0], emb)

        # exclude exact 1?
        
        scored = sorted(scored.items(), key=lambda x:-x[1]) # [(10, 0.9873), (47, blah), ..., (2, 0.0003)]
        facts = {f"sent{i+1}": self.corpus_list[pair[0]] for i, pair in enumerate(scored[:self.k])}
        # print("FACTS", facts)
        facts = " ".join([f"{k}: {facts[k]}" for k in facts]) # stringify
        # print("Hypothesis:", ex_hypothesis)
        # print(self.k, "retrieved supporting facts:", facts)

        return facts


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

class ContrieverIterativeRetriever(CustomRetriever):
    # TODO copy from score_retrieval.py 
    def __init__(self, path: str):
        assert path is not None
        # same code as dragon actually oops
        objs = []
        print(f"Initializing ContrieverIterativeRetriever")
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

class SimCSEIterativeRetriever(IterativeRetriever):

    def __init__(self, path_corpus: str, embeddings_path: str, k: int = 25):
        assert path_corpus is not None
        assert embeddings_path is not None
        print(f"Initializing SimCSE retriever, k={k}")
        self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", use_fast=True) # use_fast = True, does this do anything.
        self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        self.corpus = self.load_corpus(path_corpus)
        self.corpus_list = list(self.corpus)
        self.embeddings = torch.load(embeddings_path)
        self.k = k

    # TODO IMPLEMENT
    def top_k(self, sentences: List[str]) -> OrderedDict[str, str]:
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

# Import SimCSE models
class SimCSERetriever(CustomRetriever):

    def __init__(self, path_corpus: str, embeddings_path: str, k: int = 25):
        assert path_corpus is not None
        assert embeddings_path is not None
        print(f"Initializing SimCSE retriever, k={k}")
        self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", use_fast=True) # use_fast = True, does this do anything.
        self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        self.corpus = self.load_corpus(path_corpus)
        self.corpus_list = list(self.corpus)
        self.embeddings = torch.load(embeddings_path)
        self.k = k

    def top_k(self, example: Example) -> str: # return a context string to be passed into extract_context
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