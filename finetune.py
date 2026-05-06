from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch, re, os, random, json, argparse
import numpy as np
import wandb

def formatting(dset):
  prompts = dset["prompt"]
  return {"text": prompts}

    

def set_model(model_path): 
  if "Qwen3.5" not in model_path: 
    raise NotImplementedError(f"{model_path} is not a model that this script can call")

  model, tokenizer = FastLanguageModel.from_pretrained(
    model_path, 
    load_in_8bit=True,
    load_in_4bit=False,
    use_gradient_checkpointing = "unsloth"
  )
  tokenizer.pad_token = tokenizer.eos_token
  
  model = FastLanguageModel.get_peft_model(
    model,
    finetune_language_layers = True, 
    finetune_attention_modules = True, 
    finetune_mlp_modules = True,
    r = 8, 
    lora_alpha = 8, 
    lora_dropout = 0,
    bias = "none", 
    random_state = 42, 
    use_rslora = False,
    loftq_config = None
  )

  FastLanguageModel.for_training(model) 
  train_ds = load_dataset("AmazonScience/TISER", split="train")
  train_ds = train_ds.map(formatting, batched=True)


  trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds, 
    formatting_func = formatting,
    args = SFTConfig(
      per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 250,
       # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases
    )  
  )
  return model, tokenizer, trainer
    
  

def pull_dataset(ds_prompt): 
  if ds_prompt not in ["Focus", "Gen"]: # 'Focus' and 'Gen' are placeholders
    raise NotImplementedError("This prompt is not one of the two synthetic data prompts types")
  

def get_args(): 
  parser = argparse.ArgumentParser()

  parser.add_argument("--model_path", type=str, required=True, default=None)  # base model
 
  return parser.parse_args()
  
def main(): 
  args = get_args()
  
  seed = 42
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  pre_trained_model, tokenizer, trainer = set_model(args.model_path)

  trainer_stats = trainer.train()
  
  used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
  
  print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
  print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
  
  print(f"Peak reserved memory = {used_memory} GB.")

  save_name = f"LoRA_All"
  print(f"Model is saved to {save_name}")

  pre_trained_model = pre_trained_model.merge_and_unload()
  
  pre_trained_model.save_pretrained(save_name)
  tokenizer.save_pretrained(save_name)

  
  
  


if __name__ == "__main__": 
  main() 
