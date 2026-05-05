import os
from datasets import load_dataset
import argparse


def load_dataset(dataset_path):
  return load_dataset(dataset_path)

def get_args(): 
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_path", type=str, required=True, default=None)

return parser.parse_args()

def main(): 
  args = get_args()

  if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)
  if not os.path.exists(args.result_dir):
      os.makedirs(args.result_dir)

if __name__ == "__main__": 
  main()
