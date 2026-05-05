from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from collections import Counter
import torch
import json
import os
import argparse
import re

def get_args(): 
  parser = argparse.ArgumentParser()

  parser.add_argument("--input_dir", type=str, required=True, default=None)
  parser.add_argument("--result_dir", type=str, required=False, default="/metric_results")  # result path for evaluation

  return parser.parse_args()
  
def main():
  args = get_args()

  if not os.path.exists(args.input_dir):
    os.makedirs(args.input_dir)
  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

  
  similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

  dataset_info = {"dataset_name" : Counter(), "task" : Counter(), "level" : Counter(), "metric_type" : Counter()}
  correct_counters = {"dataset_name" : Counter(), "task" : Counter(), "level" : Counter(), "metric_type" : Counter()}
  

  with open(f"responses/{args.input_dir}.json", "r", encoding="utf-8") as file:
    dataset = json.load(file)
  
  for point in tqdm(dataset): 
    model_pred = re.sub(r'[^\w\s]', '', point["Response"])
    gold = re.sub(r'[^\w\s]', '', point["gold_answer"])

    if point["metric_type"] == "multi_choice": 
      if model_pred == gold: 
        correct_counters["dataset_name"][point["dataset_name"]] += 1
        correct_counters["task"][point["task"]] += 1
        correct_counters["level"][point["level"]] += 1
        correct_counters["metric_type"][point["metric_type"]] += 1
    else:
      encode_pred = similarity_model.encode(model_pred)
      encode_gold = similarity_model.encode(point["gold_answer"])
      similarity = similarity_model.similarity(encode_pred, encode_gold)
      
      correct_counters["dataset_name"][point["dataset_name"]] += similarity
      correct_counters["task"][point["task"]] += similarity
      correct_counters["level"][point["level"]] += similarity
      correct_counters["metric_type"][point["metric_type"]] += similarity
    
    dataset_info["dataset_name"][point["dataset_name"]] += 1
    dataset_info["task"][point["task"]] += 1
    dataset_info["level"][point["level"]] += 1
    dataset_info["metric_type"][point["metric_type"]] += 1

  accuracy_computations = {}
  
  for key in dataset_info: 
    for counter_key in dataset_info[key]: 
      acc = correct_counters[key][counter_key] / dataset_info[key][counter_key]
      accuracy_computations[f"{key}/{counter_key}"] = acc

  print("Computations Complete") 

  output_path = os.path.join(f"{args.result_dir}/{args.input_dir}_metrics.json")

  print(f"Output path for metrics is: {output_path}")
  print(accuracy_computations)
  
  with open(output_path, "w", encoding="utf-8") as f: 
    json.dump(accuracy_computations, f, ensure_ascii=False) 

if __name__ == "__main__":
  main()
