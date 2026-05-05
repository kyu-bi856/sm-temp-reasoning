from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm
import torch
import json
import os
import argparse

def load_dataset_from_path(PATH_NAME):
  if PATH_NAME not in ["TensorTemplar/TIME-Lite-Atomic"]: 
    raise NotImplementedError("<PATH_NAME> is either not implemented to allow for evaluation or it is not a valid HuggingFace dataset path.\n To know which datasets are allowed, check the README.")
  return load_dataset(PATH_NAME, split="train")

def load_model(MODEL_NAME): 
  if MODEL_NAME in ["unsloth/Qwen-3.5-0.8B", "unsloth/Qwen-3.5-2B", "unsloth/gemma-4-E2B"]:
    model, tokenizer = FastLanguageModel.from_pretrained(
      name, 
      load_in_4bit = False,
      use_gradient_checkpointing = "unsloth" 
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
    
 # if MODEL_NAME in ["Qwen800m_FT", "Qwen2B_FT", "gemmaE2B_FT"]: 
    
    
  

def extract_answer(model_output):
  print(model_output)
  return model_output[model_output.find("</think>") + 8 : ].strip("\n")

def prompt_on_TLA(model, tokenizer, sample):
  inputs = tokenizer.apply_chat_template(
      sample["prompt"],
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

  dataset_info_calculator = {"dataset_name": Counter(), "task_name": Counter(), "level_name" : Counter(), "task_type" : Counter()}
  correctness_calculator = {"dataset_name": Counter(), "task_name": Counter(), "level_name" : Counter(), "task_type" : Counter()}
  
  for point in dataset: 
    dataset_info_calculator["dataset_name"][point["dataset_name"]] += 1
    dataset_info_calculator["task_name"][point["task"]] += 1
    dataset_info_calculator["level_name"][point["level"]] += 1
    dataset_info_calculator["task_type"][point["metric_type"]] += 1



  for i in tqdm(range(len(dataset))): 
    out = prompt_on_TLA(model, tokenizer, dataset[i]) 
    
    if extract_answer(out) == dataset[i]["gold_answer"]:
      correct += 1

  print(f"\n Accuracy = {correct / len(dataset)}")
  
  

if __name__ == "__main__":
  main()

