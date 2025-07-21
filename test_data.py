import tensorflow_datasets as tfds
ds = tfds.load("warmup", data_dir="/nvme_data/zhiyuanma/RL4VLA/datasets")
for example in ds["train"].take(10):
    print(example["image"].shape)  # 检查图像形状