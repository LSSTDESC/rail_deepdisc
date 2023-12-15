import os

from deepdisc.model.models import (RedshiftPDFROIHeads,
                                   RedshiftPointCasROIHeads,
                                   RedshiftPointROIHeads)
from detectron2.config import LazyConfig, get_cfg


def get_lazy_config(cfgfile, train_head):
    cfg = LazyConfig.load(cfgfile)
    
    


    return cfg


def get_loader_config(output_dir, batch_size):
    cfg_loader = get_cfg()
    cfg_loader.SOLVER.IMS_PER_BATCH = batch_size
    cfg_loader.DATASETS.TRAIN = "astro_train"  # Register Metadata
    cfg_loader.DATASETS.TEST = "astro_val"
    # cfg_loader.DATALOADER.NUM_WORKERS = 0
    # cfg_loader.DATALOADER.PREFETCH_FACTOR = 2
    cfg_loader.SOLVER.BASE_LR = 0.001
    cfg_loader.OUTPUT_DIR = output_dir
    os.makedirs(cfg_loader.OUTPUT_DIR, exist_ok=True)

    cfg_loader.SOLVER.CLIP_GRADIENTS.ENABLED = True
    # Type of gradient clipping, currently 2 values are supported:
    # - "value": the absolute values of elements of each gradients are clipped
    # - "norm": the norm of the gradient for each parameter is clipped thus
    #   affecting all elements in the parameter
    cfg_loader.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    # Maximum absolute value used for clipping gradients
    # Floating point number p for L-p norm to be used with the "norm"
    # gradient clipping type; for L-inf, please specify .inf
    cfg_loader.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0

    cfg_loader.SOLVER.STEPS = []  # do not decay learning rate for retraining
    cfg_loader.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg_loader.SOLVER.WARMUP_ITERS = 0
    cfg_loader.TEST.DETECTIONS_PER_IMAGE = 128

    return cfg_loader
