""" This is a demo "solo config" file for use in solo_test_run_transformers.py.

This uses template configs cascade_mask_rcnn_swin_b_in21k_50ep and yaml_style_defaults."""

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling import SwinTransformer
from fvcore.common.param_scheduler import MultiStepParamScheduler
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
bs = 1

metadata = OmegaConf.create() 
metadata.classes = ["object"]

numclasses = len(metadata.classes)

# ---------------------------------------------------------------------------- #
# Standard config (this has always been the LazyConfig/.py-style config)
# ---------------------------------------------------------------------------- #
# Get values from templates
from ..common.coco_loader_lsj import dataloader
from ..COCO.cascade_mask_rcnn_swin_b_in21k_50ep import dataloader, model, train, lr_multiplier, optimizer

# Overrides
model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
dataloader.train.total_batch_size = bs
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512

dataloader.train.total_batch_size = batch_size
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512
model.backbone.bottom_up.stem.in_channels = 6
model.pixel_mean = [
        0.05381286,
        0.04986344,
        0.07526361,
        0.10420945,
        0.14229655,
        0.21245764,
]
model.pixel_std = [
        2.9318833,
        1.8443471,
        2.581817,
        3.5950038,
        4.5809164,
        7.302009,
]

model.roi_heads.num_components = 1
model.roi_heads._target_ = RedshiftPDFROIHeads
model.roi_heads.box_predictor.test_score_thresh = 0.5
model.proposal_generator.nms_thresh = 0.3

# ---------------------------------------------------------------------------- #
# Yaml-style config (was formerly saved as a .yaml file, loaded to cfg_loader)
# ---------------------------------------------------------------------------- #
# Get values from template
from .yaml_style_defaults import MISC, DATALOADER, DATASETS, GLOBAL, INPUT, MODEL, SOLVER, TEST

# Overrides
DATALOADER.PREFETCH_FACTOR = 2

DATASETS.TRAIN = "astro_train"
DATASETS.TEST = "astro_val"

SOLVER.BASE_LR = 0.001
SOLVER.CLIP_GRADIENTS.ENABLED = True
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
# Maximum absolute value used for clipping gradients
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0
SOLVER.IMS_PER_BATCH = bs