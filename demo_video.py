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
from detectron2.utils.video_visualizer import VideoVisualizer


#  model and video variables
model_name = 'X-101_RGB_60k.pth'
video_path = './output/forest_walk_1min.mp4'

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
            
    # Get one video frame 
    vcap = cv2.VideoCapture('/home/vince/Videos/forest_walk_1min.mp4')
    
    # get vcap property 
    w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    n_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # VIDEO recorder
    # Grab the stats from image1 to use for the resultant video
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')   
    # video = cv2.VideoWriter("pred_and_track_00.mp4",fourcc, 5, (w, h))  
    
    # Check if camera opened successfully
    if (vcap.isOpened()== False):
        print("Error opening video stream or file")
       
    vid_vis = VideoVisualizer(metadata=tree_metadata)
                                
    nframes = 0
    while(vcap.isOpened() ):
        ret, frame = vcap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        y = 000
        # h = 800
        x = 000
        # w = 800
        crop_frame = frame[y:y+h, x:x+w]
        # cv2.imshow('frame', crop_frame)
        if cv2.waitKey(1) == ord('q'):
                break
        
        # 5 fps
        if nframes % 12 == 0:
            outputs_pred = predictor_synth(crop_frame)
            # v_synth = Visualizer(crop_frame[:, :, ::-1],
            #                     metadata=tree_metadata, 
            #                     scale=1, 
            #                     instance_mode =  ColorMode.IMAGE     # remove color from image, better see instances  
            #     )
            out = vid_vis.draw_instance_predictions(crop_frame, outputs_pred["instances"].to("cpu"))
                
            vid_frame = out.get_image()
            # video.write(vid_frame)
            cv2.imshow('frame', vid_frame)
            
        nframes += 1
    
    # video.release()
    vcap.release()
    cv2.destroyAllWindows()
    
        
    