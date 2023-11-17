import os

from deepdisc.model.models import (RedshiftPDFROIHeads,
                                   RedshiftPointCasROIHeads,
                                   RedshiftPointROIHeads)
from detectron2.config import LazyConfig, get_cfg


def get_lazy_config(cfgfile, batch_size, numclasses):
    cfg = LazyConfig.load(cfgfile)
    bs = 1
    cfg.model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
    cfg.model.proposal_generator.batch_size_per_image = 512

    cfg.dataloader.train.total_batch_size = batch_size
    cfg.model.roi_heads.num_classes = numclasses
    cfg.model.roi_heads.batch_size_per_image = 128
    # cfg.model.backbone.bottom_up.in_chans = 6
    cfg.model.backbone.bottom_up.stem.in_channels = 6
    # cfg.model.pixel_mean = [0.00376413, 0.01292477, 0.04308919, 0.08965224, 0.13665777, 0.16544563]
    # cfg.model.pixel_std =[0.07028706, 0.06942783, 0.2432113, 0.55617297, 0.8046262, 0.9044935]
    cfg.model.pixel_mean = [
        0.00604957,
        0.0097893,
        0.02428164,
        0.04782381,
        0.06689043,
        0.07634047,
    ]
    cfg.model.pixel_std = [
        0.07637636,
        0.07277052,
        0.19012378,
        0.42937884,
        0.5813913,
        0.65684605,
    ]

    cfg.model.roi_heads.num_components = 1
    cfg.model.roi_heads._target_ = RedshiftPDFROIHeads
    # cfg.model.roi_heads._target_ = RedshiftPointROIHeads
    cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    cfg.model.proposal_generator.nms_thresh = 0.3

    return cfg


def get_loader_config(output_dir, batch_size, epochs):
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
    cfg_loader.SOLVER.MAX_ITER = epochs  # for DefaultTrainer
    cfg_loader.TEST.DETECTIONS_PER_IMAGE = 128

    return cfg_loader
