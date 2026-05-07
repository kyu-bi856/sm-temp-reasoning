# sm-temp-reasoning
for Brandeis COSI 115B Final Project

Step 1: Install Dependencies
~~~
pip install -r requirements.txt
pip install unsloth
~~~

Step 2: Finetune a Model on TISER
~~~
python finetune.py --model_path --PEFT_type --DROP
~~~
- model_path (required) can be any of the non-MoE Qwen3.5 series formatted like "unsloth/Qwen3.5-0.8B", for the 0.8B parameter model.
-  PEFT_type (optional, defaults to 16-bit LoRA) is the quantizing used for LoRA, can be either "LoRA16", "LoRA8", or "QLoRA" (for 4-bit LoRA)
-  DROP (optional, defaults to not freezing any layers) is used to indicate whether or not the Attention or MLP layers will be frozen for the purposes of finetuning. ("DropMLP" for freezing MLP layers, "DropAttn" for freezing attention layer, and "All" for freezing none of them)
Once run, the model will be saved as "{PEFT_type}_{DROP}" in the /sm-temp-reasoning/ directory and can be called by the eval.py file. 

Step 3: Evaluate Pretrained or Finetuned Model on TIME-Lite
~~~
python eval.py --model_path --dataset_path
~~~
- model_path (required) can be a string corresponding to any unsloth model ("unsloth/Qwen3.5-0.8B") or can be one of the fine-tuned models stored in the same directory ("LoRA16_All")
- dataset_path (not required, defaults to "TensorTemplar/TIME-Lite-Atomic") is the path to the HuggingFace dataset. At this current moment, eval.py only has functionality implemented for TIME-Lite-Atomic (a restructured form of "SylvainWei/TIME-Lite")
Once run, the responses will be saved to the "./responses" directory as "{model_path}.json" 

Step 4: Calculate and Display Metrics
~~~
python metrics.py --input_dir --result_dir
~~~
- input_dir (required) is the path corresponding to the responses of a particular model
- result_dir (not required, defaults to "metric_results") is the path corresponding to what directory the metrics will be placed into
Once run, the metrics will be saved as a .json file to the directory passed in and can be viewed by any methodology to view a JSON file.
Tasks that are MCQs OR Timeline tasks are evaluated on accuracy using exact match, while all other tasks are evaluated on normalized cosine similarity using $norm\_sim = \frac{1 + sim(gold, pred)}{2}$

