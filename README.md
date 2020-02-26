# phx-nn

Neural network to map summary statistics to PoolHapX parameter sets. This repository serves as the
code base for the outer model of the Deep Learning for Haplotype Reconstruction workflow. The code
here serves to:
* Split raw extracted data into training and testing sets.
* Set the neural architecture search space.
* Perform NAS through optuna.
* Train, test, and perform inference.
* Serialize the trained NN.

## Getting Started
### Prerequisites
* Python 3.7.6
* PyTorch 1.3.1
* PyTorch-Lightning 0.5.3.2
* Apex 0.1
* XGBoost 0.90
* AdaBound 0.0.5
* CudaToolKit 10.1.243
* Optuna 0.19.0
* Pandas 0.24.2
* Scikit-Learn 0.22
* Numpy 1.17.3

### Installing
Most of the packages listed can be installed through conda or pip, but NVIDIA Apex needs to be built
from source with the `--cpp_ext` flag. More information [here](https://github.com/NVIDIA/apex#quick-start).
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Running
#### Feature Extraction
To extract the summary statistics from VEF files (or BAM and VCF files), external scripts not
within this repository are applied. Copy and store a CSV file containing the data in `data/raw`.

#### Data Pre-Processing
```bash
python data_split.py <test_size> <optional_seed>
```
* `test_size`: proportion of test set for splitting dataset into training and testing sets.
* `optional_seed`: optional seed value.

#### Training
Prior to running the following, ensure that the settings and hyperparameter search spaces are
properly specified in `config.json`.
```bash
python train.py config.json <optional_seed>
```
* `optional_seed`: optional seed value.

#### Testing
```bash
python test.py config.json saved/<training_run>/training/best_arch.json
```
* `training_run`: the training run directory under `saved/`.

#### Inference
```bash
python inference.py config.json saved/<training_run>/training/best_arch.json saved/<training_run>/training/phxnn.pth <inference_sum_stats> <output>
```
* `training_run`: the training run directory under `saved/`.
* `inference_sum_stats`: the summary statistics CSV data to perform inference on.
* `output`: the output file path for the model tuned PoolHapX parameter sets.

## Built With
* [PyTorch](https://pytorch.org/) - deep learning framework.
* [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - PyTorch wrapper.
* [AdaBound](https://github.com/Luolc/AdaBound) - neural network optimizer.
* [Optuna](https://optuna.org/) - neural architecture search.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
This project is heavily inspired by the repository
[pytorch-template](https://github.com/victoresque/pytorch-template) by
[Victor Huang](https://github.com/victoresque).
