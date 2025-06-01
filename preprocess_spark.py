import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, struct
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType

from utils.general_utils import load_config
from datasets import load_from_disk


def get_spark_session(config: dict, local_spark: bool = False):
    """Initialize Spark session."""
    app_name = config['spark']['app_name_preprocess']
    builder = SparkSession.builder.appName(app_name)
    if local_spark:
        builder = builder.master("local[*]") \
                         .config("spark.driver.memory", config['spark'].get('driver_memory', "4g"))
    spark = builder.getOrCreate()
    return spark


def tokenize_and_format_udf_generator(tokenizer_name: str, max_seq_length: int):
    """Generate UDF for tokenization and formatting."""
    def tokenize_and_format_row(row):
        from transformers import AutoTokenizer

        question = row.question
        answer = row.answer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        full_text = question + " " + answer
        toks = tokenizer(full_text, padding="max_length", truncation=True,
                         max_length=max_seq_length, return_tensors=None)
        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]

        labels = list(input_ids)
        q_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
        q_len = min(len(q_ids), len(labels))
        for i in range(q_len):
            labels[i] = -100
        if tokenizer.pad_token_id is not None:
            for i, m in enumerate(attention_mask):
                if m == 0:
                    labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "original_question": question,
            "original_answer": answer
        }

    schema = StructType([
        StructField("input_ids", ArrayType(IntegerType()), False),
        StructField("attention_mask", ArrayType(IntegerType()), False),
        StructField("labels", ArrayType(IntegerType()), False),
        StructField("original_question", StringType(), True),
        StructField("original_answer", StringType(), True)
    ])
    return udf(tokenize_and_format_row, schema)


def main():
    """Spark data preprocessing."""
    parser = argparse.ArgumentParser(description="Spark Preprocessing for CI-LLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--override_config", type=str, help="Path to override config")
    parser.add_argument("--dataset_type", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--local_spark", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.override_config)
    spark = get_spark_session(config, args.local_spark)

    tokenizer_name = config['model_name']
    max_length = config['max_seq_length']
    if args.dataset_type == "train":
        output_path = config['processed_data_path_train']
        hf_split = 'train'
    else:
        output_path = config['processed_data_path_eval']
        hf_split = 'test'

    print(f"Preprocessing {args.dataset_type} → {output_path}")
    print(f"Tokenizer: {tokenizer_name}, Max length: {max_length}")

    dataset_path = config['dataset_name']
    try:
        hf_dict = load_from_disk(dataset_path)
        if hf_split not in hf_dict:
            raise ValueError(f"Split '{hf_split}' not found: {list(hf_dict.keys())}")
        hf_dataset = hf_dict[hf_split]
        print(f"Loaded split '{hf_split}', {len(hf_dataset)} examples")
        print(f"Columns: {hf_dataset.column_names}")
    except Exception as e:
        print(f"Error loading dataset at '{dataset_path}': {e}")
        raise

    if config.get("debug_subset_size") and args.dataset_type == "train":
        size = config['debug_subset_size']
        hf_dataset = hf_dataset.select(range(min(size, len(hf_dataset))))
        print(f"Debug subset applied: {len(hf_dataset)} examples")

    try:
        cols = ['question', 'answer']
        if not all(c in hf_dataset.column_names for c in cols):
            raise ValueError(f"Required columns {cols} missing: {hf_dataset.column_names}")
        pandas_df = hf_dataset.to_pandas()[cols]
        print(f"Pandas DataFrame with columns {list(pandas_df.columns)}, rows {len(pandas_df)}")
    except Exception as e:
        print(f"Error converting to Pandas or selecting columns: {e}")
        raise

    spark_schema = StructType([
        StructField("question", StringType(), True),
        StructField("answer", StringType(), True)
    ])

    raw_df = spark.createDataFrame(pandas_df, schema=spark_schema)
    print("Converted to Spark DataFrame")
    raw_df.printSchema()
    print(f"Partitions: {raw_df.rdd.getNumPartitions()}")
    raw_df.show(5, truncate=50)

    tokenizer_udf = tokenize_and_format_udf_generator(tokenizer_name, max_length)
    processed = raw_df.withColumn(
        "proc", tokenizer_udf(struct(col("question"), col("answer")))
    )

    final_df = processed.select(
        col("proc.input_ids").alias("input_ids"),
        col("proc.attention_mask").alias("attention_mask"),
        col("proc.labels").alias("labels"),
        col("proc.original_question").alias("original_question"),
        col("proc.original_answer").alias("original_answer")
    )

    final_df.printSchema()
    final_df.show(5, truncate=80)

    print(f"Writing to {output_path}")
    final_df.write.mode("overwrite").parquet(output_path)
    print("Done")
    spark.stop()


if __name__ == "__main__":
    main()
