from datasets import load_dataset

data_files = {
    "train": "recipes_train.csv",
    "validation": "recipes_val.csv",
    "test": "recipes_test.csv",
}
dataset = load_dataset("csv", data_files=data_files)

print(dataset)
# dataset["train"][0] to inspect a row
