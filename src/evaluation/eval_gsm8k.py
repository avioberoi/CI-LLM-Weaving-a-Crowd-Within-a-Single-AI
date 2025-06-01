import os
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig
from src.models.gemma_backbone import CILLMModel  
from src.aggregator.dirichlet_bayesian import DirichletAggregator


def load_config(config_path="configs/eval.yaml"):  
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    print(f"Warning: Evaluation config {config_path} not found.")

def parse_gsm8k_answer(generated_text_full: str, prompt_text: str) -> str:
    """
    Extracts the final numeric answer from the generated text for GSM8K
    The generated text includes the prompt. We need to isolate the generated part
    """
    # Remove the prompt part from the generated text
    if generated_text_full.startswith(prompt_text):
        generated_answer_part = generated_text_full[len(prompt_text):].strip()
    else:
        generated_answer_part = generated_text_full  

    if "####" in generated_answer_part:
        final_answer_text = generated_answer_part.split("####")[-1].strip()
    else:
        words = generated_answer_part.split()
        final_answer_text = words[-1] if words else ""
        print(f"Warning: '####' not found in generated answer part: '{generated_answer_part}'.")

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
    print(f"Checkpoint directory found: {checkpoint_dir}")

    # Initialize CI-LLM Model
    # It will load the base model internally and then load the PEFT adapters from the checkpoint_dir
    print(f"Initializing CILLMModel from base {config['model_name']} and PEFT checkpoint {config['trained_peft_checkpoint_dir']}")
    try:
        ci_model = CILLMModel(
            model_name=config['model_name'],
            K=config['num_agents'],  
            lora_rank=config['lora_rank'],  
            lora_alpha=config['lora_alpha'],
            lora_dropout=0.0,  
            dropconnect_p=0.0,  
            target_modules=config['lora_target_modules'],
            initialize_adapters_rand=False,  
            use_gradient_checkpointing=False,  
            trained_checkpoint_dir=config['trained_peft_checkpoint_dir'] 
        ).to(device)
    except Exception as e:
        print(f"Error initializing CILLMModel: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    ci_model.eval()  
    print("CILLMModel initialized and set to evaluation mode.")

    # Initialize Aggregator
    try:
        aggregator = DirichletAggregator(
            num_agents=ci_model.K,  
            alpha_val=config['dirichlet_alpha_aggregator'],
            input_is_logits=True  
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
        print(f"Processing only {len(test_data)} examples (subset size: {debug_size})")
    else:
        print(f"Processing full dataset: {len(test_data)} examples")
    
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
        prompt_text = example["question"]  
        
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
        with torch.no_grad():  
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
                continue  

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
