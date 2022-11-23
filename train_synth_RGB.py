#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import  absolute_import

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import torch; print(torch.__version__)
import os, json, cv2, random
import numpy as np
import time
import datetime

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils as utils
from detectron2.data.datasets.coco import load_coco_json
import detectron2.data.transforms as T
import copy

from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.engine import HookBase
import detectron2.utils.comm as comm
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine.hooks import PeriodicWriter

import albumentations as A
from pycocotools.coco import COCO, maskUtils
import logging
import pandas as pd
from tensorboard import version; print(version.VERSION)
from tqdm import tqdm
from itertools import chain


def test_mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    # This mapper uses to power of the albumentations library to optimize DA
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
    # get annotations
    bboxes = [ann['bbox'] for ann in dataset_dict['annotations']]
    labels = [ann['category_id'] for ann in dataset_dict['annotations']]
    keypoints = np.array([ann['keypoints'] for ann in dataset_dict['annotations']]).reshape((-1, 3))
    masks = [maskUtils.decode(ann['segmentation']) for ann in dataset_dict['annotations']]
    
    # FDA things
    # im_name='/home/vince/repos/coco-annotator/datasets/essai_03/image_00000_RGB.png'
    # target_image = utils.read_image(im_name, format="BGR")
    
    # Configure data augmentation -> https://albumentations.ai/docs/getting_started/transforms_and_targets/
    transform = A.Compose([
        A.RandomCrop(720, 720, p=0.0),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        bbox_params=A.BboxParams(format='coco', label_fields=['bbox_ids'], min_visibility=0.1))
    
    transformed = transform(image=image,
                            masks=masks,
                            bboxes=bboxes,
                            keypoints=keypoints,
                            category_id=labels,
                            bbox_ids=np.arange(len(bboxes)))
    
    transformed_image = transformed["image"]
    h, w, _ = transformed_image.shape
    visible_ids = transformed['bbox_ids']
    transformed_masks = [maskUtils.encode(np.asfortranarray(mask)) for mask in np.array(transformed["masks"])[visible_ids]]
    transformed_bboxes = np.array(transformed["bboxes"])
    transformed_keypoints = np.array(transformed['keypoints']).reshape((-1, 5, 3))[visible_ids]     # Ideally find a way to retrieve NUM_KEYPOINTS instead of hardcoding
    for keypoints in transformed_keypoints:
        for keypoint in keypoints:
            if keypoint[0] > w or keypoint[0] < 0 or keypoint[1] > h or keypoint[1] < 0:
                keypoint[0:2] = [-0.5, -0.5]
                keypoint[2] = 0
                
    # check if horizontal flip
    for keypoints in transformed_keypoints:
        if keypoints[1][0] > keypoints[2][0]:
            temp_kp = np.copy(keypoints[2])
            keypoints[2] = keypoints[1]
            keypoints[1] = temp_kp
    
    transformed_labels = np.array(transformed['category_id'])
    dataset_dict["image"] = torch.as_tensor(transformed_image.transpose(2, 0, 1).astype("float32"))
    annos = [
        {
            'iscrowd': 0,
            'bbox': transformed_bboxes[i].tolist(),
            'keypoints': transformed_keypoints[i].tolist(),
            'segmentation': transformed_masks[i],
            'category_id': transformed_labels[i],
            'bbox_mode': BoxMode.XYWH_ABS,
        }
        for i in range(len(transformed_bboxes))
    ]
    dataset_dict['annotations'] = annos
    instances = utils.annotations_to_instances(annos, image.shape[:2], mask_format="bitmask")
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


def albumentations_mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
    # get annotations
    bboxes = [ann['bbox'] for ann in dataset_dict['annotations']]
    labels = [ann['category_id'] for ann in dataset_dict['annotations']]
    keypoints = np.array([ann['keypoints'] for ann in dataset_dict['annotations']]).reshape((-1, 3))
    masks = [maskUtils.decode(ann['segmentation']) for ann in dataset_dict['annotations']]
        
    # Configure data augmentation -> https://albumentations.ai/docs/getting_started/transforms_and_targets/
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(720, 720, p=1.0),
        A.RandomBrightnessContrast(p=0.3, brightness_limit=[-0.1, 0.1], contrast_limit=[-0.1, 0.3], brightness_by_max=True),
        
        A.GaussNoise(p=0.2, var_limit=(10.0, 50.0), mean=0, per_channel=True),
        A.GlassBlur(p=0.1, sigma=0.6, max_delta=3, iterations=2, mode='fast'),
        A.ISONoise(p=0.2, color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
                        
        A.HueSaturationValue(p=0.3, sat_shift_limit=0.25, hue_shift_limit=0, val_shift_limit=0),
        A.MotionBlur(p=0.2, blur_limit=7),
        A.Perspective(p=0.2),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        bbox_params=A.BboxParams(format='coco', label_fields=['bbox_ids'], min_visibility=0.1))
    
    transformed = transform(image=image,
                            masks=masks,
                            bboxes=bboxes,
                            keypoints=keypoints,
                            category_id=labels,
                            bbox_ids=np.arange(len(bboxes)))
    
    transformed_image = transformed["image"]
    h, w, _ = transformed_image.shape
    visible_ids = transformed['bbox_ids']
    transformed_masks = [maskUtils.encode(np.asfortranarray(mask)) for mask in np.array(transformed["masks"])[visible_ids]]
    transformed_bboxes = np.array(transformed["bboxes"])
    transformed_keypoints = np.array(transformed['keypoints']).reshape((-1, 5, 3))[visible_ids]     # Ideally find a way to retrieve NUM_KEYPOINTS instead of hardcoding
    for keypoints in transformed_keypoints:
        for keypoint in keypoints:
            if keypoint[0] > w or keypoint[0] < 0 or keypoint[1] > h or keypoint[1] < 0:
                keypoint[0:2] = [-0.5, -0.5]
                keypoint[2] = 0
                
    # check if horizontal flip
    for keypoints in transformed_keypoints:
        if keypoints[1][0] > keypoints[2][0]:
            temp_kp = np.copy(keypoints[2])
            keypoints[2] = keypoints[1]
            keypoints[1] = temp_kp
    
    transformed_labels = np.array(transformed['category_id'])
    dataset_dict["image"] = torch.as_tensor(transformed_image.transpose(2, 0, 1).astype("float32"))
    annos = [
        {
            'iscrowd': 0,
            'bbox': transformed_bboxes[i].tolist(),
            'keypoints': transformed_keypoints[i].tolist(),
            'segmentation': transformed_masks[i],
            'category_id': transformed_labels[i],
            'bbox_mode': BoxMode.XYWH_ABS,
        }
        for i in range(len(transformed_bboxes))
    ]
    dataset_dict['annotations'] = annos
    instances = utils.annotations_to_instances(annos, image.shape[:2], mask_format="bitmask")
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict



# https://github.com/facebookresearch/detectron2/issues/1763
# https://gilberttanner.com/blog/detectron-2-object-detection-with-pytorch
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=albumentations_mapper
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=test_mapper
        )    

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, ("bbox", "segm", "keypoints"), False, output_dir=output_folder, kpt_oks_sigmas=(.25, .25, .25, .25, .25))    # ("bbox", "segm", "keypoints")

    def build_hooks(self):
        hooks = super(MyTrainer, self).build_hooks()
        cfg = self.cfg
        if len(cfg.DATASETS.TEST) > 0:
            loss_eval_hook = LossEvalHook(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                MyTrainer.build_test_loader(cfg, cfg.DATASETS.TEST[0]),
            )
            hooks.insert(-1, loss_eval_hook)

        return hooks


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        # self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        # return losses
        return mean_loss

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = int(self.trainer.iter) + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            mean_loss = self._do_loss_eval()
            self.trainer.storage.put_scalars(validation_loss=mean_loss)
            print("validation do loss eval", mean_loss)
        else:
            pass
         
# name of the .pth file
model_name = 'your-coco-pretrained-weights.pth'

img_dir = 'path/to/synthtree/images'

if __name__ == "__main__":
    
    torch.cuda.is_available()
    
    coco_train_filename='./output/train_RGB.json'
    coco_val_filename='./output/val_RGB.json'
    coco_test_filename='./output/test_RGB.json'
    
    train_dataset_name="tree_train_set"
    val_dataset_name="tree_val_set"
    test_dataset_name="tree_test_set"
    
    logger = setup_logger(name=__name__)    
    
    dicts_train = load_coco_json(coco_train_filename, img_dir, train_dataset_name)  
    logger.info("Done loading {} samples.".format(len(dicts_train)))
    dicts_val = load_coco_json(coco_val_filename, img_dir, val_dataset_name)  
    logger.info("Done loading {} samples.".format(len(dicts_val)))
    dicts_test = load_coco_json(coco_test_filename, img_dir, test_dataset_name)  
    logger.info("Done loading {} samples.".format(len(dicts_test)))
    
    for d in ["train_set"]:
        DatasetCatalog.register("tree_" + d, lambda d=d: dicts_train)
        MetadataCatalog.get("tree_" + d).set(thing_classes=["tree"], keypoint_names=["kpCP", "kpL", "kpR", "ax1", "ax2"], keypoint_flip_map=[])
        
    for d in ["val_set"]:
        DatasetCatalog.register("tree_" + d, lambda d=d: dicts_val)
        MetadataCatalog.get("tree_" + d).set(thing_classes=["tree"], keypoint_names=["kpCP", "kpL", "kpR", "ax1", "ax2"], keypoint_flip_map=[])
    
    for d in ["test_set"]:
        DatasetCatalog.register("tree_" + d, lambda d=d: dicts_test)
        MetadataCatalog.get("tree_" + d).set(thing_classes=["tree"], keypoint_names=["kpCP", "kpL", "kpR", "ax1", "ax2"], keypoint_flip_map=[])
    
    
    cfg = get_cfg()
    # cfg = LazyConfig.load(model_zoo.get_config_file("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_list(opts)
    cfg.DATASETS.TRAIN = ("tree_train_set",)
    cfg.DATASETS.VAL = ("tree_val_set",)
    cfg.DATASETS.TEST = ("tree_test_set",)
    cfg.DATALOADER.NUM_WORKERS = 8
    # better to load the weigths from a COCO model rather than a COCO-keypoint model
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.SOLVER.IMS_PER_BATCH = 4    # 8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = [10000, 30000]
    cfg.SOLVER.BASE_LR = 0.002  # pick a good LR
    cfg.SOLVER.MAX_ITER = 60000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (tree)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1  
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
    cfg.TEST.KEYPOINT_OKS_SIGMAS = (.25, .25, .25, .25, .25)
    cfgMODEL.BACKBONE.FREEZE_AT = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.TEST.EVAL_PERIOD = 2000    # only uncomment when evaluating during training
    cfg.INPUT.MIN_SIZE_TEST = 0  # no resize at test time
    
    cfg.CUDNN_BENCHMARK = True
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.KEYPOINT_ON = True   
    cfg.OUTPUT_DIR = './output'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    metrics_df = pd.read_json(cfg.OUTPUT_DIR + "/metrics.json", orient="records", lines=True)
    mdf = metrics_df.sort_values("iteration")
    # print(mdf)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    # cfg.INPUT.MIN_SIZE_TEST = 0  # no resize at test time
    
    predictor_synth = DefaultPredictor(cfg)
    
    dir_fold_test = cfg.OUTPUT_DIR + "/eval_0"
    os.makedirs(dir_fold_test, exist_ok=True)
    evaluator = COCOEvaluator("tree_test_set", cfg, False, output_dir=dir_fold_test)
    val_loader = build_detection_test_loader(cfg, "tree_test_set")
    print(inference_on_dataset(predictor_synth.model, val_loader, evaluator))
        
    
    # visualize detections
    dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TEST]))
    random.shuffle(dicts)
    tree_metadata = MetadataCatalog.get("tree_val_set")
    for dic in tqdm(dicts):
        img = utils.read_image(dic["file_name"], "BGR")
        outputs_synth = predictor_synth(img)
        v_synth = Visualizer(img[:, :, ::-1],
                        metadata=tree_metadata, 
                        scale=1, 
                        instance_mode =  ColorMode.IMAGE     # remove color from image, better see instances  
        )
        
        # remove keypoints
        # outputs_synth["instances"].remove('pred_keypoints')
        
        out_synth = v_synth.draw_instance_predictions(outputs_synth["instances"].to("cpu"))
        
        cv2.imshow('predictions', out_synth.get_image()[:, :, ::-1])
        # cv2.imshow('predictions', img)
        k = cv2.waitKey(0)
        
        # exit loop if esc is pressed
        if k == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
