# data_processing/preprocess_spark.py
import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, struct
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, MapType

# Transformers needs to be available on the Spark workers if UDFs use it directly.
# This is handled by --py-files in the sbatch script for project modules,
# and for libraries like transformers, it should be in the anaconda-2022.05 environment.
# from transformers import AutoTokenizer # Moved to UDF scope for safety

from utils.general_utils import load_config # Assuming this can be found via PYTHONPATH
from datasets import load_from_disk # <<< IMPORT ADDED

def get_spark_session(config: dict, local_spark: bool = False):
    """Initializes and returns a Spark session."""
    app_name = config['spark']['app_name_preprocess']
    builder = SparkSession.builder.appName(app_name)

    if local_spark:
        print("Configuring Spark for local execution.")
        builder = builder.master("local[*]") \
                         .config("spark.driver.memory", config['spark'].get('driver_memory', "4g"))
    else:
        print("Spark will run based on spark-submit parameters (YARN/cluster mode).")
        # On a cluster, master, executor memory, etc., are set by spark-submit in sbatch.

    spark = builder.getOrCreate()
    return spark


# This UDF will be applied to each row of the DataFrame
# It needs access to AutoTokenizer.
def tokenize_and_format_udf_generator(tokenizer_name, max_seq_length):
    # This outer function is called once on the driver to create the UDF.
    # The inner function is what gets serialized and run on workers.
    def tokenize_and_format_row(row_data):
        # Import here to ensure it's available on workers
        from transformers import AutoTokenizer

        question = row_data.question
        answer = row_data.answer # This includes reasoning and final answer for GSM8K

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        full_text = question + " " + answer
        
        tokenized_inputs = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors=None # Return Python lists/ints
        )
        
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        
        # Create labels: mask the question part
        labels = list(input_ids) # Make a mutable copy
        question_token_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
        question_len = len(question_token_ids)
        actual_question_mask_len = min(question_len, len(labels))
        
        for i in range(actual_question_mask_len):
            labels[i] = -100 # Mask question tokens

        # Mask padding tokens in labels if pad_token_id is known
        if tokenizer.pad_token_id is not None:
            for i in range(len(labels)):
                if attention_mask[i] == 0: # This assumes attention_mask[i] == 0 for padding
                    labels[i] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "original_question": question, # Keep for reference if needed
            "original_answer": answer
        }
    
    # Define the schema for the output of the UDF
    # This must match the structure of the dictionary returned by tokenize_and_format_row
    output_schema = StructType([
        StructField("input_ids", ArrayType(IntegerType()), False),
        StructField("attention_mask", ArrayType(IntegerType()), False),
        StructField("labels", ArrayType(IntegerType()), False),
        StructField("original_question", StringType(), True),
        StructField("original_answer", StringType(), True)
    ])
    
    return udf(tokenize_and_format_row, output_schema)

def main():
    parser = argparse.ArgumentParser(description="Spark Data Preprocessing for CI-LLM")
    parser.add_argument("--config", type=str, required=True, help="Path to the main configuration file.")
    parser.add_argument("--override_config", type=str, help="Path to an override configuration file.")
    parser.add_argument("--dataset_type", type=str, choices=["train", "eval"], default="train", 
                        help="Which dataset split to process: train or eval.")
    parser.add_argument("--local_spark", action="store_true", help="Run Spark in local mode.")
    args = parser.parse_args()

    config = load_config(args.config, args.override_config)
    spark = get_spark_session(config, args.local_spark)

    tokenizer_name = config['model_name']
    max_length = config['max_seq_length']
    
    if args.dataset_type == "train":
        # input_path = config['raw_data_path'] # Assuming raw_data_path is for training - REMOVED
        output_path = config['processed_data_path_train']
        dataset_hf_split = 'train' # For load_dataset if used as a source
    else: # eval
        # You might have a different raw path for eval data, or use a split from the same raw path
        # input_path = config.get('raw_data_path_eval', config['raw_data_path']) - REMOVED
        output_path = config['processed_data_path_eval']
        dataset_hf_split = 'test' # For load_dataset if used as a source

    print(f"Starting Spark preprocessing for dataset: {args.dataset_type}")
    # print(f"Input path: {input_path}") # REMOVED
    print(f"Output path: {output_path}")
    print(f"Tokenizer: {tokenizer_name}, Max length: {max_length}")

    # --- Data Loading ---
    # Option 1: Load from a JSONL file as specified in your example config
    # Assumes each line is a JSON object with "question" and "answer" fields.
    # Spark schema needs to be defined or inferred.
    # raw_df_schema = StructType([
    #     StructField("question", StringType(), True),
    #     StructField("answer", StringType(), True) # Ensure this contains the full GSM8K answer string
    # ])
    # raw_df = spark.read.schema(raw_df_schema).json(input_path)
    
    # Option 2: Load from Hugging Face datasets (if running locally or driver has access and libs)
    # This is often easier for standard datasets like GSM8K.
    # Note: This loads data on the DRIVER, then distributes. For truly massive files on HDFS,
    # loading directly with spark.read.format is better.
    # from datasets import load_dataset as hf_load_dataset # <<< REMOVED
    
    dataset_path = config['dataset_name'] # Should be "data/hf_datasets/gsm8k_main"
    
    print(f"Loading Hugging Face dataset dictionary using load_from_disk from path '{dataset_path}' on driver...")
    try:
        # Use load_from_disk as per user suggestion
        full_hf_dataset_dict = load_from_disk(dataset_path)
        
        print(f"Available splits in loaded dataset: {list(full_hf_dataset_dict.keys())}")
        print(f"Dataset structure: {full_hf_dataset_dict}") # Log the structure
        
        if dataset_hf_split not in full_hf_dataset_dict:
            raise ValueError(f"Requested split '{dataset_hf_split}' not found in loaded dataset. Available: {list(full_hf_dataset_dict.keys())}")
            
        hf_dataset = full_hf_dataset_dict[dataset_hf_split]
        print(f"Successfully loaded split '{dataset_hf_split}'. Number of examples: {len(hf_dataset)}")
        print(f"Column names in loaded split: {hf_dataset.column_names}")

    except Exception as e:
        print(f"Error loading dataset using load_from_disk from path '{dataset_path}' (split: '{dataset_hf_split}'): {e}")
        raise # Re-raise the exception to stop the script

    if config.get("debug_subset_size") and args.dataset_type == "train": # Apply debug subset for train only
        debug_size = config['debug_subset_size']
        print(f"Applying debug subset of size: {debug_size}")
        hf_dataset = hf_dataset.select(range(min(debug_size, len(hf_dataset)))) # Ensure not to select more than available
        print(f"Number of examples after debug subset selection: {len(hf_dataset)}")
    
    # --- Column Selection (should not be strictly necessary if load_from_disk works as expected) ---
    # However, we will explicitly select to be sure and to match user's suggestion for the pandas df
    print(f"Attempting to select 'question' and 'answer' columns from loaded split '{dataset_hf_split}'...")
    try:
        required_columns = ['question', 'answer']
        available_columns = hf_dataset.column_names
        
        if not all(col in available_columns for col in required_columns):
            raise ValueError(f"CRITICAL: Not all required columns ({required_columns}) found in dataset. Available: {available_columns}. Dataset loading might be incorrect.")

        # Select only 'question' and 'answer' for the pandas DataFrame
        pandas_df = hf_dataset.to_pandas()[required_columns]
        print(f"Successfully created Pandas DataFrame with columns: {list(pandas_df.columns)}. Number of examples: {len(pandas_df)}")

    except Exception as e:
        print(f"Error during column selection or Pandas conversion: {e}")
        raise

    # Define explicit Spark schema as per user suggestion
    spark_schema = StructType([
        StructField("question", StringType(), True),
        StructField("answer",   StringType(), True),
    ])

    print("Pandas DataFrame (with 'question', 'answer' columns only) Info:")
    pandas_df.info()
    
    # Create Spark DataFrame with explicit schema
    raw_df = spark.createDataFrame(pandas_df, schema=spark_schema)
    print("Converted Hugging Face dataset to Spark DataFrame with explicit schema.")

    raw_df.printSchema()
    print(f"Number of partitions in raw_df: {raw_df.rdd.getNumPartitions()}")
    raw_df.show(5, truncate=50)

    # --- Tokenization and Formatting ---
    # Create the UDF instance
    tokenizer_udf = tokenize_and_format_udf_generator(tokenizer_name, max_length)

    # Apply the UDF. The UDF returns a struct, which will be a new column.
    # We then select the fields from this struct into their own columns.
    # Pass a struct of the necessary columns to the UDF
    processed_df_with_struct = raw_df.withColumn(
        "processed_data", 
        tokenizer_udf(struct(col("question"), col("answer")))
    )
    
    final_df = processed_df_with_struct.select(
        col("processed_data.input_ids").alias("input_ids"),
        col("processed_data.attention_mask").alias("attention_mask"),
        col("processed_data.labels").alias("labels"),
        col("processed_data.original_question").alias("original_question"),
        col("processed_data.original_answer").alias("original_answer")
    )

    final_df.printSchema()
    print(f"Showing sample of tokenized data (first 5 rows):")
    final_df.show(5, truncate=80)

    # --- Saving Processed Data ---
    print(f"Saving processed data to Parquet: {output_path}")
    # Ensure the output directory parent exists if Spark doesn't create it (usually does)
    # os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    final_df.write.mode("overwrite").parquet(output_path)

    print(f"Successfully processed and saved data to {output_path}")
    spark.stop()

if __name__ == "__main__":
    main() 