## openvla in rl4vla vs official

#### 删掉了：
```bash
experiments
Makefile
.pre-commit-config.yaml
scripts
vla-scripts/extern
```

#### 新增了：
```bash
rlds_dataset_builder
vla-scripts/merge_lora.py
```

#### 修改了：

`prismatic/extern/hf/modeling_prismatic.py` 里面新增了出Value的操作和类和attention_mask的处理 \
`prismatic/extern/hf/processing_prismatic.py` 里面把图片改成用tensor表示，新增了attention_mask的处理 \
`prismatic/vla/datasets/datasets.py` 加了俩参数 \
`prismatic/vla/datasets/rlds/dataset.py` 加了俩参数 \
`prismatic/vla/datasets/rlds/oxe/configs.py` 里面加上了warmup和sft的数据配置 \
`prismatic/vla/datasets/rlds/oxe/transforms.py` 里面加上了warmup和sft的transform \
`vla-scripts/finetune.py` 只用lora，修改了wandb的一些输出，如eval等 


## official_openvla vs openvla-oft:

#### oft新增：
```bash
vla-scripts/merge_lora_weights_and_save.py
prismatic/vla/constants.py
prismatic/training/train_utils.py
prismatic/models/projectors.py
prismatic/models/film_vit_wrapper.py
prismatic/models/action_heads.py
```

#### oft修改：
`vla-scripts/finetune.py` 
`vla-scripts/deploy.py` 
`prismatic/vla/datasets/rlds/utils/data_utils.py` 
`prismatic/vla/datasets/rlds/traj_transforms.py` 
`prismatic/vla/datasets/rlds/oxe/transforms.py` 
`prismatic/vla/datasets/rlds/oxe/mixtures.py` 
`prismatic/vla/datasets/rlds/oxe/materialize.py` 
`prismatic/vla/datasets/rlds/oxe/configs.py` 
`prismatic/vla/datasets/rlds/dataset.py` 
`prismatic/vla/datasets/datasets.py` 
`prismatic/util/data_utils.py` 
`prismatic/training/strategies/base_strategy.py` 
`prismatic/extern/hf/modeling_prismatic.py` 