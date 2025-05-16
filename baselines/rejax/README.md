# 🚀 How to run baselines using Rejax

1. 📦 Install requirements from `pyproject.toml` with `pip install .`
2. ⚙️ Run `python train.py --config <path-to-config> --algorithm <algorithm-name>`
To reproduce results from the paper, use `--num-seeds 32`. Wandb logging is available with `--use-wandb`. For more information, use `python train.py --help`.

## 🗂️ Configs
All configs used for benchmarks are in the `configs` folder:
- `configs/navix_crossings_s9n1.yaml`
- `configs/navix_distshift2_1m.yaml`
- `configs/navix_doorkey_8x8_1m.yaml`
- `configs/navix_dynamic_obstacles_6x6_random_1m.yaml`
- `configs/navix_empty_5x5_1m.yaml`
- `configs/navix_empty_6x6_1m.yaml`
- `configs/navix_empty_16x16_1m.yaml`
- `configs/navix_empty_random_8x8_1m.yaml`
- `configs/navix_fourrooms_1m.yaml`
- `configs/navix_gotodoor_6x6_1m.yaml`
- `configs/navix_lavagap_s6_1m.yaml`

## Algorithms 
Choose one of:
- `ppo`
- `dqn`
- `sac`

or add you own configs for any of the algorithms in [Rejax](https://github.com/kerajli/rejax).
