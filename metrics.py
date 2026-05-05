from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import json
import os
import argparse

def get_args(): 
  parser = argparse.ArgumentParser()

  parser.add_argument("--dataset_path", type=str, required=True, default=None)
  parser.add_argument("--result_dir", type=str, required=False, default="../metric_results")  # result path for evaluation

  return parser.parse_args()
  
def main():
  args = get_args()
  


  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

  
  similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



#  metrics(dataset, args) 

      
 #   if dataset[i]["metric_type"] == "multi_choice":
 #     if extract_answer(out) == dataset[i]["gold_answer"]:
 #       correctness_calculator["dataset_name"][dataset[i]["dataset_name"]] += 1
 #       correctness_calculator["task_name"][dataset[i]["task"]] += 1
 #       correctness_calculator["level_name"][dataset[i]["level"]] += 1
 #       correctness_calculator["task_type"][dataset[i]["metric_type"]] += 1
 #   else:
 #     encode_answer = model.encode(extract_answer(out))
 #     encode_gold = model.encode(dataset[i]["gold_answer"])
#      similarity = model.similarity(encode_answer, encode_gold)

  #    correctness_calculator["dataset_name"][dataset[i]["dataset_name"]] += similarity
   #   correctness_calculator["task_name"][dataset[i]["task"]] += similarity
   #   correctness_calculator["level_name"][dataset[i]["level"]] += similarity
   #   correctness_calculator["task_type"][dataset[i]["metric_type"]] += similarity


 #dataset_info_calculator = {"dataset_name": Counter(), "task_name": Counter(), "level_name" : Counter(), "task_type" : Counter()}
#  correctness_calculator = {"dataset_name": Counter(), "task_name": Counter(), "level_name" : Counter(), "task_type" : Counter()}
  
  #for point in dataset: 
  #  dataset_info_calculator["dataset_name"][point["dataset_name"]] += 1
 #   dataset_info_calculator["task_name"][point["task"]] += 1
 #   dataset_info_calculator["level_name"][point["level"]] += 1
 #   dataset_info_calculator["task_type"][point["metric_type"]] += 1
  
  

if __name__ == "__main__":
  main()
