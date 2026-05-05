from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm
import torch
import json
import os
import argparse
import random
import numpy as np
import anthropic

def get_args(): 
  parser = argparse.ArgumentParser()

  parser.add_argument("--prompt_type", type=str, required=True, default=None)
  parser.add_argument("--output_dir", type=str, required=False, default="./synth_data")  # output path for evaluation

  return parser.parse_args()
  
def main():
  args = get_args()
  
  seed = 42
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  client = anthropic.Anthropic() 

  message = client.messages.create(
    model="claude-opus-4.6", 
    max_tokens = 1000, 
    messages = [
      {  
        "role" : "user", 
        "content": prompt
    ]
  )
  message.content


    

  




if __name__ == "__main__":
  main()
