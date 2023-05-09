import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import json
from embed import load_corpus_sentences
from common import normalize, extract_context
import numpy as np
import sys 
import re
import string


def compute_hyp_embeddings(hf_tokenizer: str, hf_model: str, hyps, model_type: str, verbose=False):
    ''' Returns a tensor of the hypotheses embeddings '''
    if verbose:
        print("Downloading models")
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer) 
    model = AutoModel.from_pretrained(hf_model)
    embeds = None
    if model_type=="simcse":
        hyps_tok = tokenizer(hyps, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeds = model(**hyps_tok, output_hidden_states=True, return_dict=True).pooler_output
    elif model_type=="dragon":
        hyps_tok = tokenizer(hyps, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeds = model(**hyps_tok).last_hidden_state[:, 0, :]
    elif model_type=="contriever":
        hyp_inp = tokenizer(hyps, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            hyp_emb_out = model(**hyp_inp)
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
        embeds = mean_pooling(hyp_emb_out[0], hyp_inp['attention_mask'])
    else:
        print("Unrecognized model type.")
        return
    return embeds


##############################################

def hyp_greedy_retrieve(hyp_embeds, corpus_embeds, topk, verbose=False):
    scored_list = []
    for j, h in enumerate(hyp_embeds):
        if verbose and j % 10 == 0:
            print("Starting retrieval for hypothesis", j)
        scored = {i: 1-cosine(h, emb) for i, emb in enumerate(corpus_embeds)}
        scored_list.append(sorted(scored.items(), key=lambda x:-x[1])[:topk])
    return scored_list

def conditional_retrieve(hyps, hyp_embeds, corpus, corpus_embeds, topk, w: int, verbose=False):
    # DEPRECATED: due to poor performance, we do not use this method in reporting results, etc
    ''' w ≤ topk: omega parameter in Ribeiro conditional retrieval algorithm '''
    scored_list = []
    for j, h in enumerate(hyp_embeds):
        if verbose and j % 10 == 0:
            print("Starting retrieval for hypothesis", j)
        # first do same process as greedy_retrieve for first w of topk premises
        scored = {i: 1-cosine(h, emb) for i, emb in enumerate(corpus_embeds)}
        scored = sorted(scored.items(), key=lambda x:-x[1])[:w]

        # for the remainder: do iterative/cumulative "conditional" retrieval (modified from Ribeiro pseudocode)
        qset = set([corpus[x[0]] for x in scored])
        cumu_emb_sims = np.array([sum([1-cosine(corpus_embeds[x[0]], emb) for x in scored]) for emb in corpus_embeds])
        most_recent_add = h
        qset.add(hyps[j])
        for _ in range(topk-w): 
            best_i, best_i_score = None, None
            for ind, emb in enumerate(corpus_embeds):
                if corpus[ind] in qset:
                    continue 
                cumu_emb_sims[ind] += 1-cosine(most_recent_add, emb)
                if best_i_score is None or cumu_emb_sims[ind] > best_i_score:
                    best_i = ind 
                    best_i_score = cumu_emb_sims[ind]
            most_recent_add = corpus_embeds[best_i]
            qset.add(corpus[best_i])
            scored.append((best_i, best_i_score))
        scored_list.append(scored)
    return scored_list

def split_hyp_retrieval(hyp_embeds, half_hyp_embeds, corpus_embeds, topk, each_half_k, verbose=False):
    ''' half_hyp_embeds: should be a tensor of (# hypotheses x 2 x embedding_dimension) size'''
    scored_list=[]
    full_hyp_n = topk-2*each_half_k
    for j, h in enumerate(hyp_embeds):
        if verbose and j % 10 == 0:
            print("Starting retrieval for hypothesis", j)
        scored, first_half_scored, second_half_scored = {}, {}, {}
        for i, emb in enumerate(corpus_embeds):
            scored[i] = 1-cosine(h, emb)
            first_half_scored[i] = 1-cosine(half_hyp_embeds[j][0], emb)
            second_half_scored[i] = 1-cosine(half_hyp_embeds[j][1], emb)
        
        scored_list.append(sorted(scored.items(), key=lambda x:-x[1])[:full_hyp_n])
        first_half_scored = sorted(first_half_scored.items(), key=lambda x:-x[1])
        second_half_scored = sorted(second_half_scored.items(), key=lambda x:-x[1])

        ind=0
        while(len(scored_list[-1]) < full_hyp_n+each_half_k):
            if first_half_scored[ind][0] not in [y[0] for y in scored_list[-1]]:
                scored_list[-1].append(first_half_scored[ind])
            ind+=1
        ind=0
        while(len(scored_list[-1]) < topk):
            if second_half_scored[ind][0] not in [y[0] for y in scored_list[-1]]:
                scored_list[-1].append(second_half_scored[ind])
            ind+=1
    return scored_list

##############################################

def hyp_embedtype_helper(hyps, model_type, verbose):
    if model_type=="simcse":
        bert_model = "princeton-nlp/sup-simcse-bert-base-uncased"
        hyp_embeds = compute_hyp_embeddings(bert_model, bert_model, hyps, model_type, verbose=verbose)
    elif model_type=="contriever":
        model='facebook/contriever'
        hyp_embeds = compute_hyp_embeddings(model, model, hyps, model_type, verbose=verbose)
    elif model_type=="dragon":
        model='facebook/dragon-roberta-query-encoder'
        hyp_embeds = compute_hyp_embeddings(model, model, hyps, model_type, verbose=verbose)
    else:
        print("Unknown model type.")
        return
    return hyp_embeds

def retrieve_context(task3_path: str, corpus_embedding_path: str, output_json_path: str, topk: int, model_type: str, algo="greedy", w=None, write_to_file=True, verbose=False):
    hyps = []
    if verbose:
        print("Reading hypothesis from", task3_path)
    for line in open(task3_path):
        ex = json.loads(line)
        hyps.append(normalize(ex["hypothesis"]))
    ###
    # hyps = hyps[:4] # uncomment to use a reduced set of hypotheses just for testing
    ###
    if verbose:
        print("Computing hypotheses embeddings")
    hyp_embeds = hyp_embedtype_helper(hyps, model_type, verbose)
    # hyp_embeds = torch.load("simcse_hntrain_hypembeds.pt")
    if verbose:
        print("Loading precomputed tensor from", corpus_embedding_path)
    corpus_embeddings = torch.load(corpus_embedding_path)
    if verbose:
        print("Loading corpus sentences")
    corpus = load_corpus_sentences("data/entailment_trees_emnlp2021_data_v3/supporting_data/worldtree_corpus_sentences_extended.json")
    
    ##############################
    if verbose:
        print("Computing contexts for each hypothesis")
    context = []
    if algo=="greedy": # do standard greedy retrieval
        scored_list = hyp_greedy_retrieve(hyp_embeds, corpus_embeddings, topk, verbose=verbose)
    elif algo=="conditional": #do conditional retrieval
        scored_list = conditional_retrieve(hyps, hyp_embeds, corpus, corpus_embeddings, topk, w, verbose=verbose)
    elif algo=="split": # do split-hypothesis greedy retrieval
        split_hyps = []
        for h in hyps:
            words = h.split()
            first_half_sz = len(words)//2
            split_hyps.append(' '.join(words[:first_half_sz]))
            split_hyps.append(' '.join(words[first_half_sz:]))
        split_hyp_embeds = hyp_embedtype_helper(split_hyps, model_type, verbose)
        split_hyp_embeds = torch.reshape(split_hyp_embeds, (len(hyps), 2, len(hyp_embeds[0])))
        scored_list = split_hyp_retrieval(hyp_embeds, split_hyp_embeds, corpus_embeddings, topk, each_half_k=w, verbose=verbose)
    ##############################
    for j, scored in enumerate(scored_list):
        context.append({"hypothesis":hyps[j], "context":[corpus[i[0]] for i in scored]})

    if write_to_file:
        if verbose:
            print("Writing hypothesis and context to JSON file")
        with open(output_json_path, 'w') as fout:
            for d in context:
                json.dump(d, fout)
                fout.write("\n")

    return hyps, context


##############################################
# these methods taken from EntailmentBank code: utils/proof_utils
def squad_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_sentences(sentences, normalize_fn=None):
    if isinstance(sentences, str):
        print("here")
        # Streams of non-period characters starting with non-space.
        sentences = re.findall("[^. ][^.]+\\.", sentences)
    if normalize_fn is not None:
        sentences = [normalize_fn(s) for s in sentences]
    return sentences

# Score P/R/F1 for sentence overlaps, e.g., for RuleTaker inferences
def score_sentence_overlaps(sentences, sentences_gold, normalize_fn=None):
    sentences = normalize_sentences(sentences, normalize_fn)
    sentences_gold = normalize_sentences(sentences_gold, normalize_fn)
    # print(f"\t\tsentences:{sentences}")
    # print(f"\t\tsentences_gold:{sentences_gold}")
    if len(sentences) == 0 or len(sentences_gold) == 0:
        if sentences == sentences_gold:
            prec = recall = 1
        else:
            prec = recall = 0
    else:
        common = len(set(sentences).intersection(set(sentences_gold)))
        # Duplicates in sentences are penalized in precision
        prec = common / len(sentences)
        recall = common / len(sentences_gold)
    if prec == recall == 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    acc = 0 if f1 < 1 else 1
    # if acc < 1.0:
    #     print(f"\n\n+++++++++++++++++++++++++")
    #     print(f"[score_sentence_overlaps]sentences:{sentences}")
    #     print(f"[score_sentence_overlaps]sentences_gold:{sentences_gold}")
    #     print(f"[score_sentence_overlaps]{acc}\t{f1}")
    return {"P": prec, "R": recall, "F1": f1, "acc": acc, "pred": sentences, "gold": sentences_gold}
##############################################

def score_all_context(pred_context_path: str, gold_data_path: str, score_file_path: str, save_to_file=True, is_dalvi=False):
    pred_contexts = []
    for line in open(pred_context_path): 
        ex = json.loads(line)
        if is_dalvi: # from task3 dataset
            pred_context = list(extract_context(ex["context"]).values())
        else: # our predicted context, from our json file (different format)
            pred_context = ex["context"] 
        pred_contexts.append(sorted(pred_context))
    gold_contexts = []
    for line in open(gold_data_path): # should be in same order 
        ex = json.loads(line)
        gold_context = list(extract_context(ex["context"]).values())
        gold_contexts.append(sorted(gold_context))
    scores = [score_sentence_overlaps(pred_contexts[i], gold_contexts[i], normalize_fn=squad_normalize_answer) for i in range(len(pred_contexts))]
    if save_to_file:
        with open(score_file_path, 'w') as fout:
            for d in scores:
                json.dump(d, fout)
                fout.write("\n")
    return scores 

def avg_recall(scores, verbose=False):
    return np.mean([score_dict["R"] for score_dict in scores])

def main():

    DEV = False # set this variable to True if want to use dev paths, false otherwise (for test paths)
    if DEV: 
        task3_path = "data/entailment_trees_emnlp2021_data_v3/dataset/task_3/dev.jsonl" 
        gold_path = "../entailment_bank/data/processed_data/slots/task_1-slots/dev.jsonl"
    else: 
        task3_path = "data/entailment_trees_emnlp2021_data_v3/dataset/task_3/test.jsonl"
        gold_path = "../entailment_bank/data/processed_data/slots/task_1-slots/test.jsonl"

    ##############################################
    # score dalvi contexts
    if sys.argv[1]=="score" and sys.argv[2]=="dalvi":
        pred_path = "data/entailment_trees_emnlp2021_data_v3/dataset/task_3/test.jsonl"
        score_path = "test_res/dalvi_retrieval_test_scores.json"
        dalvi_scores = score_all_context(pred_path, gold_path, score_path, save_to_file=False, is_dalvi=True)
        print("R@25 for Dalvi 25:", avg_recall(dalvi_scores))
    ##############################################
    # score simcse contexts
    if sys.argv[1]=="score" and sys.argv[2]=="simcse":
        pred_path = "test_res/simcse_split_task3_test_contexts.json"
        score_path = "test_res/simcse_split_test_scores.json"
        simcse_scores = score_all_context(pred_path, gold_path, score_path, save_to_file=False)
        print("R@25 for Split SimCSE (Test):", avg_recall(simcse_scores))
    ##############################################
    # score contriever contexts
    # use DEV paths 
    if sys.argv[1]=="score" and sys.argv[2]=="contriever":
        pred_path = "test_res/contriever_split_task3_test_contexts.json"
        score_path = "test_res/contriever_split_test_scores.json"
        contriever_scores = score_all_context(pred_path, gold_path, score_path, save_to_file=False)
        print("R@25 for Split Contriever (Test):", avg_recall(contriever_scores))
    ##############################################
    # score dragon contexts
    if sys.argv[1]=="score" and sys.argv[2]=="dragon":
        # pred_path = "dragon_splitv2_task3_dev_contexts.json"
        pred_path = "test_res/dragon_split_task3_test_contexts.json"
        score_path = "test_res/dragon_split_test_scores.json"
        dragon_scores = score_all_context(pred_path, gold_path, score_path, save_to_file=False)
        print("R@25 for Split DRAGON (Test):", avg_recall(dragon_scores))
    ##############################################
    # compute simcse contexts
    if sys.argv[1]=="context" and sys.argv[2]=="simcse":
        corpus_emb_path = "wtcorpus_simcse_embeddings_full.pt"
        # output_json_path = "simcse_split_task3_dev_contexts.json"
        output_json_path = "test_res/simcse_split_task3_test_contexts.json"
        hyps, context = retrieve_context(task3_path, corpus_emb_path, output_json_path, topk=25, model_type="simcse", algo="split", w=5, write_to_file=True, verbose=True)
    ##############################################
    # compute contriever contexts
    if sys.argv[1]=="context" and sys.argv[2]=="contriever":
        corpus_emb_path = "wtcorpus_contriever_embeddings_full.pt"
        # output_json_path = "contriever_split_task3_dev_contexts.json"
        output_json_path = "test_res/contriever_split_task3_test_contexts.json"
        hyps, context = retrieve_context(task3_path, corpus_emb_path, output_json_path, topk=25, model_type="contriever", algo="split", w=5, write_to_file=True, verbose=True)
    ##############################################
    # compute dragon contexts
    if sys.argv[1]=="context" and sys.argv[2]=="dragon":
        corpus_emb_path = "wtcorpus_dragon_embeddings_full.pt"
        output_json_path = "test_res/dragon_split_task3_test_contexts.json"
        hyps, context = retrieve_context(task3_path, corpus_emb_path, output_json_path, topk=25, model_type="dragon", algo="split", w=5, write_to_file=False, verbose=True)
    ##############################################


if __name__ == "__main__":
    main()