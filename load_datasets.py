import os
import argparse


def get_args(): 
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_name", type=str, required=True, default=None)

return parser.parse_args()

def main(): 
  args = get_args()

  if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)
  if not os.path.exists(args.result_dir):
      os.makedirs(args.result_dir)

if __name__ == "__main__": 
  main()
