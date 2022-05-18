#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test trained network on a video
"""
from __future__ import  absolute_import

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, cv2
import torch

# import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer


# local paths to model and image
model_name = 'X-101_RGB_60k.pth'
image_path = './output/image_00000_RGB.png'

if __name__ == "__main__":
    torch.cuda.is_available()
    logger = setup_logger(name=__name__)
    
    # All configurables are listed in /repos/detectron2/detectron2/config/defaults.py        
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (tree)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1  
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
    cfg.MODEL.MASK_ON = True
    
    cfg.OUTPUT_DIR = './output'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # cfg.INPUT.MIN_SIZE_TEST = 0  # no resize at test time
    
    # set detector
    predictor_synth = DefaultPredictor(cfg)    
    
    # set metadata
    tree_metadata = MetadataCatalog.get("my_tree_dataset").set(thing_classes=["Tree"], keypoint_names=["kpCP", "kpL", "kpR", "AX1", "AX2"])
    
    # inference
    im = cv2.imread(image_path)
    outputs_pred = predictor_synth(im)
    v_synth = Visualizer(im[:, :, ::-1],
                    metadata=tree_metadata, 
                    scale=1,
    )
    out_synth = v_synth.draw_instance_predictions(outputs_pred["instances"].to("cpu"))
    cv2.imshow('predictions', out_synth.get_image()[:, :, ::-1])
    k = cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
        