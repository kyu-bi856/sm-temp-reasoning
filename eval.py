from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm
import torch
import json
import os
import argparse

def load_TIME_LITE():
  return load_dataset("TensorTemplar/TIME-Lite-Atomic", split="train")

def load_TIME(): 
  return load_dataset("SylvainWei/TIME")

def load_model(name): 
  model, tokenizer = FastLanguageModel.from_pretrained(
    name, 
    load_in_4bit = False,
    use_gradient_checkpointing = "unsloth" 
  )
  tokenizer.pad_token = tokenizer.eos_token
  return model, tokenizer

def extract_answer(model_output):
  return model_output[model_output.find("</think>") + 8 : ].strip("\n")

def prompt_on_TLA(model, tokenizer, sample):
  messages = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : sample["context"]},
            {"type" : "text", "text" : "\nAnswer the following question with only the letter corresponding to the correct answer:\n"},
            {"type" : "text", "text" : sample["question"]},
            ],
        },
    ]

  inputs = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      tokenize=True,
      padding=True,
      return_tensors="pt",
  ).to("cuda")

  output = model.generate(
    inputs,
    max_new_tokens=16,
    pad_token_id=tokenizer.pad_token_id
)
  return tokenizer.decode(output[0], skip_special_tokens=True)

def get_args(): 
  parser = argparse.ArgumentParser()

  parser.add_argument("--model_path", type=str, required=False, default=None)  # models for evaluation
#  parser.add_argument("--dataset_path", type=str, required=True, default=None)
  parser.add_argument("--output_dir", type=str, required=False, default="../responses")  # output path for evaluation
  parser.add_argument("--result_dir", type=str, required=False, default="../metric_results")  # result path for evaluation

  return parser.parse_args()
  
def main():
  args = get_args()
  correct = 0

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

  model, tokenizer = load_model("unsloth/Qwen3.5-0.8B")
  dataset = load_TIME_LITE()

  for i in tqdm(range(len(dataset))): 
    out = prompt_on_TLA(model, tokenizer, dataset[i]) 
    if extract_answer(out) == dataset[i]["gold_answer"]:
      correct += 1

  print(f"\n Accuracy = {correct / len(dataset)}")
  
  

if __name__ == "__main__":
  main()

