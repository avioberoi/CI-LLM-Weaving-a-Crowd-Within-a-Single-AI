import os
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig
from src.models.gemma_backbone import CILLMModel # Assuming CILLMModel is updated as discussed
from src.aggregator.dirichlet_bayesian import DirichletAggregator
# DataLoader is not strictly needed for GSM8K eval if processing one by one, but useful for batching
# from torch.utils.data import DataLoader 

def load_config(config_path="configs/eval.yaml"): # Separate eval config if desired
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    print(f"Warning: Evaluation config {config_path} not found. Using default eval config values")

def parse_gsm8k_answer(generated_text_full: str, prompt_text: str) -> str:
    """
    Extracts the final numeric answer from the generated text for GSM8K
    The generated text includes the prompt. We need to isolate the generated part
    """
    # Remove the prompt part from the generated text
    if generated_text_full.startswith(prompt_text):
        generated_answer_part = generated_text_full[len(prompt_text):].strip()
    else:
        # Fallback if prompt isn't exactly at the start (e.g. due to special tokens)
        # This might need more robust handling based on tokenizer behavior
        generated_answer_part = generated_text_full # Assume it's mostly answer if prompt not found

    # Standard GSM8K answer parsing: text after "####"
    if "####" in generated_answer_part:
        final_answer_text = generated_answer_part.split("####")[-1].strip()
    else:
        # Fallback: try to extract a number from the end of the string if "####" is missing
        # This is less reliable and often indicates a malformed generation.
        words = generated_answer_part.split()
        final_answer_text = words[-1] if words else ""
        print(f"Warning: '####' not found in generated answer part: '{generated_answer_part}'. Using last word: '{final_answer_text}'")

    # Clean the extracted numeric string (remove commas, units, etc.)
    # Keep only digits, decimal point, and negative sign.
    cleaned_answer = "".join(filter(lambda char: char.isdigit() or char == '.' or char == '-', final_answer_text))
    
    # Remove trailing dots if any (e.g. "123.")
    if cleaned_answer.endswith('.'):
        cleaned_answer = cleaned_answer[:-1]
        
    return cleaned_answer.strip()

def get_gold_answer(example_answer_field: str) -> str:
    """Extracts the gold numeric answer from the GSM8K dataset's answer field."""
    gold_str = example_answer_field.split("####")[-1].strip()
    # Clean it similarly to the predicted answer for fair comparison
    cleaned_gold = "".join(filter(lambda char: char.isdigit() or char == '.' or char == '-', gold_str))
    if cleaned_gold.endswith('.'):
        cleaned_gold = cleaned_gold[:-1]
    return cleaned_gold.strip()

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Tokenizer
    print(f"Loading tokenizer for {config['model_name']}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config['model_name'], 
            trust_remote_code=True,
            local_files_only=True  # Force local loading only, no internet access
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Make sure the tokenizer is cached locally in your HF_HOME directory.")
        raise e
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer pad_token was None, set to eos_token: {tokenizer.eos_token}")

    # Verify checkpoint exists
    checkpoint_dir = config['trained_peft_checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    print(f"✓ Checkpoint directory found: {checkpoint_dir}")

    # Initialize CI-LLM Model
    # It will load the base model internally and then load the PEFT adapters from the checkpoint_dir
    print(f"Initializing CILLMModel from base {config['model_name']} and PEFT checkpoint {config['trained_peft_checkpoint_dir']}")
    try:
        ci_model = CILLMModel(
            model_name=config['model_name'],
            K=config['num_agents'], # This K should ideally match the K from the loaded checkpoint
            lora_rank=config['lora_rank'], # These LoRA params are for structure if checkpoint_dir is None
            lora_alpha=config['lora_alpha'],
            lora_dropout=0.0, # Dropout is off during eval
            dropconnect_p=0.0, # DropConnect is off during eval
            target_modules=config['lora_target_modules'],
            initialize_adapters_rand=False, # Not relevant when loading
            use_gradient_checkpointing=False, # Not relevant for eval
            trained_checkpoint_dir=config['trained_peft_checkpoint_dir'] # Key for loading
        ).to(device)
    except Exception as e:
        print(f"Error initializing CILLMModel: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    ci_model.eval() # Set model to evaluation mode
    print("CILLMModel initialized and set to evaluation mode.")

    # Initialize Aggregator
    try:
        aggregator = DirichletAggregator(
            num_agents=ci_model.K, # Use K from the model, which might be adjusted if loaded from checkpoint
            alpha_val=config['dirichlet_alpha_aggregator'],
            input_is_logits=True # CILLMModel.forward returns logits
        ).to(device)
        aggregator.eval()
        print("DirichletAggregator initialized.")
    except Exception as e:
        print(f"Error initializing DirichletAggregator: {e}")
        raise e

    # Load GSM8K Test Dataset
    print(f"Loading dataset: {config['dataset_name']}, subset: {config['dataset_subset']} (test split)")
    dataset = load_dataset(config['dataset_name'], config['dataset_subset'])
    test_data = dataset["test"]
    
    # Apply debug subset if enabled
    if config.get('debug_mode', False):
        debug_size = config.get('debug_subset_size', 100)
        test_data = test_data.select(range(min(debug_size, len(test_data))))
        print(f"🔧 DEBUG MODE: Processing only {len(test_data)} examples (subset size: {debug_size})")
    else:
        print(f"Processing full dataset: {len(test_data)} examples")
    
    # test_data = test_data.select(range(20)) # For quick testing of the eval loop

    # Generation Configuration
    # Ensure pad_token_id is set in the generation_config for the model
    # CILLMModel's generate method will try to use self.peft_model.generation_config
    # or create a default one. We can also pass one explicitly.
    generation_config = GenerationConfig(
        max_new_tokens=config['max_new_tokens'],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Common settings for greedy decoding for GSM8K
        do_sample=False,
        num_beams=1,
    )
    print(f"Generation config: max_new_tokens={generation_config.max_new_tokens}")

    correct_predictions = 0
    total_examples = 0
    results_log = []

    print(f"Starting evaluation on {len(test_data)} examples...")
    for i, example in enumerate(test_data):
        prompt_text = example["question"] # GSM8K prompts are just the questions
        
        # Tokenize the prompt
        inputs = tokenizer(
            prompt_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=config['max_seq_length_for_prompt']
        ).to(device)

        if inputs.input_ids.shape[1] >= config['max_seq_length_for_prompt']:
            print(f"Warning: Prompt for example {i} was truncated.")
        
        full_generated_ids = None
        with torch.no_grad(): # Ensure no gradients are computed during inference
            try:
                full_generated_ids = ci_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=generation_config,
                    aggregator=aggregator
                )
            except Exception as e:
                print(f"Error during generation for example {i} ('{prompt_text[:50]}...'): {e}")
                results_log.append({
                    "index": i, "question": prompt_text, "gold_answer_raw": example["answer"],
                    "generated_text": "ERROR_DURING_GENERATION", "predicted_answer_parsed": "ERROR",
                    "gold_answer_parsed": get_gold_answer(example["answer"]), "correct": False
                })
                continue # Skip to next example

        # Decode the full generated sequence (prompt + answer)
        # The generate method in CILLMModel returns the full sequence including input_ids
        full_generated_text = tokenizer.decode(full_generated_ids[0], skip_special_tokens=True)
        
        predicted_answer_parsed = parse_gsm8k_answer(full_generated_text, prompt_text)
        gold_answer_parsed = get_gold_answer(example["answer"])

        is_correct = False
        if predicted_answer_parsed == gold_answer_parsed:
            correct_predictions += 1
            is_correct = True
        
        total_examples += 1

        results_log.append({
            "index": i,
            "question": prompt_text,
            "gold_answer_raw": example["answer"],
            "generated_text_full": full_generated_text,
            "generated_answer_part": full_generated_text[len(prompt_text):].strip() if full_generated_text.startswith(prompt_text) else "N/A",
            "predicted_answer_parsed": predicted_answer_parsed,
            "gold_answer_parsed": gold_answer_parsed,
            "correct": is_correct
        })

        if (i + 1) % 10 == 0 or i == len(test_data) - 1:
            current_accuracy = (correct_predictions / total_examples * 100) if total_examples > 0 else 0
            print(f"Processed {total_examples}/{len(test_data)} examples. "
                  f"Current Accuracy: {current_accuracy:.2f}% ({correct_predictions}/{total_examples})")

    final_accuracy = (correct_predictions / total_examples * 100) if total_examples > 0 else 0
    print(f"\nGSM8K Evaluation Complete.")
    print(f"Final Accuracy: {final_accuracy:.2f}% ({correct_predictions}/{total_examples})")

    # Save detailed results
    if config.get("output_file"):
        import json
        output_filepath = os.path.join(config.get("output_dir", "."), config["output_file"])
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "w") as f:
            for record in results_log:
                f.write(json.dumps(record) + "\n")
        print(f"Detailed results saved to {output_filepath}")

if __name__ == "__main__":
    main()
