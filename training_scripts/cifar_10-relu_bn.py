from src.modeling.vgg import COB_VGG_Config, COB_VGG
from src.training.train import train
from src.training.config import TrainingConfig


## Create model config

vgg_config = COB_VGG_Config(
    ## Architecture
    num_classes=10,
    norm="batchnorm",
    pool_type="maxpool",
    act_func_type="elementwise",
    act_func_name="relu",
    smaller_32x32_classifier_in=True,
    ## Training
    init_strategy="kaiming_uniform",
)

training_config = TrainingConfig(
    ## Metadata
    name="vgg16_relu_bn",
    notes=None,
    ## Dataset
    dataset_name="cifar-10",
    ## Training
    lr=1e-1,
    lr_warmup_steps=0,
    weight_decay=5e-4,
    num_train_steps=int(12e4),
    batch_size=1024,
    ## Optimizer
    optimizer_name="sgd",
    optimizer_kwargs={"momentum": 0.9, "nesterov": True},
    ## Eval / logging
    log_interval=10,
    eval_interval=200,
)


###------------------------------------###
""" No touchy-touchy beyond this point """
###------------------------------------###


model = COB_VGG(vgg_config)


train(training_config, model)
