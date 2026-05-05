from unsloth import FastLanguageModel, UnslothLanguageDataCollator
from trl import SFTTrainer, SFTConfig
import torch, re, os, random, json, argparse
import numpy as np
import wandb

def set_model(model_path): 
  if "Qwen3.5" not in model_path: 
    raise NotImplementedError(f"{model_path} is not a model that this script can call")

  model, tokenizer = FastLanguageModel.from_pretrained(
    model_path, 
    load_in_4bit=False,
    use_gradient_checkpointing = "unsloth"
  )
  model = FastLanguageModel.get_peft_model(
    model,
    finetune_language_layers = True, 
    finetune_attention_modules = True, 
    finetune_mlp_modules = True,
    r = 16, 
    lora_alpha = 16, 
    lora_alpha = 0, 
    bias = "none", 
    random_state = 42, 
    use_rslora = False,
    loftq_config = None
  )

  FastVisionModel.for_training(model) 

  trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothLanguageCollator(model, tokenizer), 
    train_dataset = PLACEHOLDER, 
    args = SFTConfig(
      per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
       # max_steps = 30,
        num_train_epochs = 1, # Set this instead of max_steps for full training runs
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
  parser.add_argument("--synth_prompt_dataset", type=str, required=True, default=None)
 
  return parser.parse_args()
  
def main(): 
  args = get_args()
  
  seed = 42
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  synth_ft_dataset = pull_dataset(args.synth_prompt_dataset)
  pre_trained_model, tokenizer, trainer = set_model(args.model_path)

  trainer_stats = trainer.train()
  
  used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
  used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
  used_percentage = round(used_memory / max_memory * 100, 3)
  lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
  
  print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
  print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
  
  print(f"Peak reserved memory = {used_memory} GB.")
  print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
  print(f"Peak reserved memory % of max memory = {used_percentage} %.")
  print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

  save_name = f"{args.model_path.replaceall("unsloth/", "")}_{args.synth_prompt_dataset.replaceall("synth_data/", "")}"
  
  model.save_pretrained(save_name)
  tokenizer.save_pretrained(save_name)

  
  
  


if __name__ == "__main__": 
  main() 
