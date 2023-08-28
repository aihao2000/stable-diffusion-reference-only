from datasets import load_dataset

trin_dataset = load_dataset(
    "/home/aihao/workspace/DeepLearningContent/datasets/images",
    data_dir="/home/aihao/workspace/DeepLearningContent/datasets/images",
    name="reference_only_for_automatic_coloring",
)
