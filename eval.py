from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import json
import os
import argparse

def load_dataset_from_path(PATH_NAME):
  if PATH_NAME not in ["TensorTemplar/TIME-Lite-Atomic"]: 
    raise NotImplementedError("<PATH_NAME> is either not implemented to allow for evaluation or it is not a valid HuggingFace dataset path.\n To know which datasets are allowed, check the README.")
  return load_dataset(PATH_NAME, split="train")

def load_model(MODEL_NAME): 
  if MODEL_NAME in ["unsloth/Qwen3.5-0.8B", "unsloth/Qwen3.5-2B", "unsloth/gemma-4-E2B"]:
    model, tokenizer = FastLanguageModel.from_pretrained(
      MODEL_NAME, 
      load_in_4bit = False,
      use_gradient_checkpointing = "unsloth" 
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
    
 # if MODEL_NAME in ["Qwen800m_FT", "Qwen2B_FT", "gemmaE2B_FT"]: 

def prompt(sample): 
  return [
    {
      "role" : "user",
      "content": [
        {"type":"text", "text": sample["prompt"]}
      ]
    }
  ]

def prompt_on_TLA(model, tokenizer, sample):
  inputs = tokenizer.apply_chat_template(
      prompt(sample),
      add_generation_prompt=True,
      tokenize=True,
      padding=True,
      return_tensors="pt",
  ).to("cuda")

  output = model.generate(
    inputs,
    max_new_tokens=1024,
    pad_token_id=tokenizer.pad_token_id
)
  return tokenizer.decode(output[0], skip_special_tokens=True)

def get_args(): 
  parser = argparse.ArgumentParser()

  parser.add_argument("--model_path", type=str, required=True, default=None)  # models for evaluation
  parser.add_argument("--dataset_path", type=str, required=True, default=None)
  parser.add_argument("--output_dir", type=str, required=False, default="../responses")  # output path for evaluation
  parser.add_argument("--result_dir", type=str, required=False, default="../metric_results")  # result path for evaluation

  return parser.parse_args()
  
def main():
  args = get_args()

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

  model, tokenizer = load_model(args.model_path)
  dataset = load_dataset_from_path(args.dataset_path)

  all_outs = []
  
  for i in tqdm(range(len(dataset))): 
    out = prompt_on_TLA(model, tokenizer, dataset[i]) 
    all_outs.append(out)
    print(out)

  for data, out in zip(dataset, all_outs):
    data["Response"] = out

  output_path = os.path.join(args.output_dir, f"{args.model_path.split('/')[-1]}_{args.dataset_name}_{args.task}.json")

  with open(output_path, "w", encoding="utf-8") as f: 
    json.dump(dataset, f, ensure_ascii=False) 

#  metrics(dataset, args) 


def metrics(dataset, args): 
  similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

  
  
      
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

