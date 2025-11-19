# Change-of-Basis (CoB) Pruning via Rotational Invariance

### Environment

Install:
```bash
conda env create -f environment.yml && conda activate cob-pruning
```

### Training

Training scripts for TSRA w/ RMSNorm and ReLU w/ BatchNorm models on CIFAR-10 are available in training_scripts.

To run:
```python
python training_scripts/cifar_10-relu_bn.py
```

And
```python
python training_scripts/cifar_10-tsra_rms.py
```

### Download pretrained weights

From project root, create folder for model weights and enter:
```bash
mkdir pretrained_weights && cd pretrained_weights
```

Download model weights from Hugging Face and untar:
```bash
wget https://huggingface.co/Lapisbird/cob-pruning-models/resolve/main/cifar_10-relu_bn.tar.gz https://huggingface.co/Lapisbird/cob-pruning-models/resolve/main/cifar_10-tsra_rms.tar.gz
tar -xzf cifar_10-relu_bn.tar.gz && tar -xzf cifar_10-tsra_rms.tar.gz
rm cifar_10-relu_bn.tar.gz cifar_10-tsra_rms.tar.gz
```

Downloaded weights can be loaded into a model via COB_VGG's .from_pretrained(...) method. Just provide the directory which contains the pretrained model's config.json and model_state_dict.pth:
```python
from src.modeling.vgg import COB_VGG
from src.utils import get_project_root

model = COB_VGG.from_pretrained(get_project_root() / "pretrained_weights" / "cifar_10-relu_bn")
```

### Evaluate and Prune

You can experiment with evaluation and pruning in the provided Python notebook located at notebooks/prune_playground.ipynb