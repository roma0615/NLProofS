# goal: take a results file and output a jsonl file in the necessary format for evaluation
import sys
import os
import json
from common import normalize_sentence

if len(sys.argv) != 3:
  print("Usage: python results_to_jsonl.py <val/test> <version_folder_path>")
  sys.exit(1)

split = sys.argv[1]
if split not in ("val", "test"):
    print("Split must be 'val' or 'test'")
    sys.exit(1)

splitd = "dev" if split == "val" else split
# read gold to get hypothesis to ID mapping
# gold_path = f"../../entailment_bank/data/processed_data/slots/task_1-slots/{splitd}.jsonl"
gold_path = f"../../entailment_bank/data/processed_data/slots/task_2-slots/{splitd}.jsonl"
with open(gold_path, "r") as f:
  golds = [json.loads(line) for line in f.readlines()]
  hyp_to_id = {g["hypothesis"]: g["id"] for g in golds}

# read worldtree dataset to get sentence to uuid mapping
worldtree_path = "../data/entailment_trees_emnlp2021_data_v3/supporting_data/worldtree_corpus_sentences_extended.json"
with open(worldtree_path) as f:
  corpus = json.load(f)
sent_to_uuid = {normalize_sentence(v): k for k, v in corpus.items()}
# print(list(sent_to_uuid.items())[:10])
# print(sent_to_uuid["soil erosion is a kind of erosion"])

version_folder = sys.argv[2]
res_json = os.path.join(version_folder, f"results_{split}.json")
with open(res_json, "r") as f:
  proofs = json.load(f)

s = ''

for i, proof in enumerate(proofs):
  # proof: proof_pred, score, hypothesis, context
  # print(proof.keys())
  
  obj = {
    "id": hyp_to_id[proof["hypothesis"]],
    "slots": {
      "proof": proof["proof_pred"]
    },
    "worldtree_provenance": {
      sentid: {
        "uuid": sent_to_uuid[proof["context"][sentid]],
        "original_text": proof["context"][sentid]
      } for sentid in proof["context"]
    }
  }

  s += json.dumps(obj) + "\n"

output_file = os.path.join(version_folder, f"results_{split}.jsonl")
with open(output_file, "w") as f:
  f.write(s)

print("Written to", output_file)
