"""
Build a Detectron2 CfgNode for Faster R-CNN (ResNet-50 + FPN) configured
exactly as described in Section 4.1 of the paper.

Key changes from the COCO defaults:
  Anchor sizes      : [8, 16, 32, 64, 128]  (was [32,64,128,256,512])
  RPN batch size    : 512                    (was 256)
  RPN pos. fraction : 0.8                    (was 0.5)
  LR                : 0.001
  Batch size        : 4
"""

import os

from detectron2 import model_zoo
from detectron2.config import get_cfg

import data_prep.config as config


def build_cfg(
    num_epochs:      int   = config.NUM_EPOCHS,
    num_train_imgs:  int   = 0,       # set at runtime after prepare_data
    output_dir:      str   = config.CKPT_DIR,
    eval_period:     int   = 0,       # 0 → calculated as 1 epoch
    use_pretrained:  bool  = True,
) -> "CfgNode":
    """
    Return a fully-configured CfgNode.

    Args:
        num_epochs      : total training epochs (100 per paper)
        num_train_imgs  : number of training crops; used to convert epochs → iterations
        output_dir      : where checkpoints and logs are written
        eval_period     : evaluation interval in iterations (0 = every epoch)
        use_pretrained  : initialise backbone from COCO pre-trained weights
    """
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    cfg.DATASETS.TRAIN = ("waterfowl_train",)
    cfg.DATASETS.TEST  = ("waterfowl_val",)

    # ── Backbone weights ──────────────────────────────────────────────────────
    if use_pretrained:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    else:
        cfg.MODEL.WEIGHTS = ""

    # ── Anchor generator (Section 4.1) ────────────────────────────────────────
    cfg.MODEL.ANCHOR_GENERATOR.SIZES          = [[8], [16], [32], [64], [128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS  = [[0.5, 1.0, 2.0]] * 5

    # ── RPN (Section 4.1) ─────────────────────────────────────────────────────
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE  = config.RPN_BATCH_SIZE_PER_IMAGE   # 512
    cfg.MODEL.RPN.POSITIVE_FRACTION     = config.RPN_POSITIVE_FRACTION      # 0.8
    cfg.MODEL.RPN.IOU_THRESHOLDS        = [config.RPN_BG_IOU_THRESH,
                                            config.RPN_FG_IOU_THRESH]       # [0.3, 0.7]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN    = config.RPN_PRE_NMS_TOP_N_TRAIN
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN   = config.RPN_POST_NMS_TOP_N_TRAIN
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST     = config.RPN_PRE_NMS_TOP_N_TEST
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST    = config.RPN_POST_NMS_TOP_N_TEST

    # ── ROI head ──────────────────────────────────────────────────────────────
    cfg.MODEL.ROI_HEADS.NUM_CLASSES          = 1    # Detectron2 excludes background
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST    = config.CONF_THRESHOLD

    # ── Input size ────────────────────────────────────────────────────────────
    cfg.INPUT.MIN_SIZE_TRAIN = (config.CROP_SIZE,)
    cfg.INPUT.MAX_SIZE_TRAIN = config.CROP_SIZE
    cfg.INPUT.MIN_SIZE_TEST  = config.CROP_SIZE
    cfg.INPUT.MAX_SIZE_TEST  = config.CROP_SIZE

    # ── Solver (Section 4.1) ──────────────────────────────────────────────────
    cfg.SOLVER.IMS_PER_BATCH = config.BATCH_SIZE   # 4
    cfg.SOLVER.BASE_LR       = config.LEARNING_RATE  # 0.001
    cfg.SOLVER.MOMENTUM      = 0.9
    cfg.SOLVER.WEIGHT_DECAY  = 1e-4

    # Convert epochs → iterations
    if num_train_imgs > 0:
        iters_per_epoch = max(1, num_train_imgs // config.BATCH_SIZE)
    else:
        iters_per_epoch = 1000   # placeholder; updated in main when count is known
    max_iter = num_epochs * iters_per_epoch
    cfg.SOLVER.MAX_ITER       = max_iter
    cfg.SOLVER.STEPS          = (int(max_iter * 0.7), int(max_iter * 0.9))
    cfg.SOLVER.GAMMA          = 0.1
    cfg.SOLVER.WARMUP_ITERS   = min(1000, iters_per_epoch)
    cfg.SOLVER.CHECKPOINT_PERIOD = iters_per_epoch * 5  # save every 5 epochs

    # ── Evaluation ────────────────────────────────────────────────────────────
    cfg.TEST.EVAL_PERIOD = eval_period if eval_period > 0 else iters_per_epoch

    # ── Dataloader ────────────────────────────────────────────────────────────
    cfg.DATALOADER.NUM_WORKERS = config.NUM_WORKERS

    # ── Output ────────────────────────────────────────────────────────────────
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)

    return cfg
