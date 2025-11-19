from src.modeling.vgg import COB_VGG_Config, COB_VGG
from src.training.train import train
from src.training.config import TrainingConfig


## Create model config

vgg_config = COB_VGG_Config(
    ## Architecture
    num_classes=10,
    norm="rmsnorm",
    norm_kwargs={"elementwise_affine": False},
    pool_type="avgpool",
    act_func_type="tsra",
    act_func_name="logistic",
    smaller_32x32_classifier_in=True,
)

## Create training config
training_config = TrainingConfig(
    ## Metadata
    name="vgg16_tsra_rms",
    notes=None,
    ## Dataset
    dataset_name="cifar-10",
    ## Training
    lr=1e-3,
    weight_decay=5e-4,
    num_train_steps=int(12e4),
    batch_size=1024,
    ## Eval / logging
    log_interval=10,
    eval_interval=200,
)


###------------------------------------###
""" No touchy-touchy beyond this point """
###------------------------------------###


model = COB_VGG(vgg_config)


train(training_config, model)
