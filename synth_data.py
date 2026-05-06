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




"""
MCQ
Read the following context and answer the question. 
Context: {CONTEXT}
Question: {QUESTION}
Please select only ONE correct option from the choices given above. Provide only the letter of the correct option (e.g., "A").

MCQ-Ordering

FREEFORM
Read the following context and answer the question.
Context: {CONTEXT}
Question: {QUESTION} (Hint: Please answer in the form of Month Day, Year. e.g. 1 year 2 months 3days, or 2 days, or 9 months, or 3 months 15 days.)
Please provide a concise answer to the question based on the context. Your answer should be as brief as possible while still being accurate and complete. If the answer is a number, provide only the number. If it's a date, provide only the date. If it's a name, provide only the name. Avoid unnecessary explanations.


WIKI
Read the following context and answer the question. 
Context: {CONTEXT}
Question: {QUESTION}
Please provide a concise answer to the question based on the context. If the question cannot be answered based on the given context, respond with "There is no answer." Your answer should be as brief as possible while still being accurate.
      
    """
  elif prompt_type == "SPEC": 
    return f"""
    """
  
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
  
  n_questions = 270

  subtopics_prompt = """
  Given a topic, generate a list of 11 subtopics that are related to the topic. 
  The topic is: Temporal Reasoning
  The list must be without numbers, and without any descriptions of the subtopics. 
  The subtopics should be separated by commas. There must be no other text besides the list.
  """

  subtopics = client.messages.create(
  model="claude-opus-4.6", 
  max_tokens = 1024, 
  messages = [
      {  
        "role" : "user", 
        "content": subtopics_prompt
    ],
  temperature=1
  ).content

  print(subtopics)
  


    

  




if __name__ == "__main__":
  main()
