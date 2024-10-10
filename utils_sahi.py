# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset.

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlpackage          # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

SLICE_H = 640 # any multiple by 32 number
SLICE_W = 640
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2
VERBOSE_SAHI = 0

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode
from validator import BaseValidator

from yolov5 import DetectionValidator, DetectionPredictor
from sahi.slicing import get_slice_bboxes
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class DetectionValidator_SAHI(BaseValidator):
    def __init__(self, args=None, data="data/coco128.yaml", model=None, dataloader=None):
        super().__init__(data, args)
        self.args = args
        self.callbacks = Callbacks()
        self.model = model
        self.dataloader = dataloader
        self.save_dir = Path("")

    def run_model(self, plots=True):
        """
        Run the model with the current options.

        Args:
            plots (bool): Whether or not to generate plots.
        """
        # Here the actual model inference logic goes, for example:
        # model = torch.load(self.weights[0])
        # result = model(img)
        LOGGER.info("Model running with the following options:")
        for k, v in self.args.items():
            LOGGER.info(f"{k}: {v}")

        # Initialize/load model and set device
        training = self.model is not None
        if training:  # called by train.py
            device, pt, jit, engine = next(self.model.parameters()).device, True, False, False  # get model device, PyTorch model
            half &= device.type != "cpu"  # half precision only supported on CUDA
            self.model.half() if half else self.model.float()
        else:  # called directly
            device = select_device(self.args.device, batch_size=batch_size)

            # Directories
            save_dir = increment_path(Path(self.args.project) / self.args.name, exist_ok=self.args.exist_ok)  # increment run
            (save_dir / "labels" if self.args.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Load model
            model = DetectMultiBackend(self.args.weights, device=device, dnn=self.args.dnn, data=self.args.data, fp16=self.args.half)
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_img_size(self.args.imgsz, s=stride)  # check image size
            half = model.fp16  # FP16 supported on limited backends with CUDA
            if engine:
                batch_size = model.batch_size
            else:
                device = model.device
                if not (pt or jit):
                    batch_size = 1  # export.py models default to batch-size 1
                    LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

            # Data
            data = check_dataset(data)  # check

        # Configure
        model.eval()
        cuda = device.type != "cpu"
        is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
        nc = 1 if self.args.single_cls else int(data["nc"])  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        # Dataloader
        if not training:
            if pt and not self.args.single_cls:  # check --weights are trained on --data
                ncm = model.model.nc
                assert ncm == nc, (
                    f"{self.args.weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                    f"classes). Pass correct combination of --weights and --data that are trained together."
                )
            model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
            pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
            task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
            dataloader = create_dataloader(
                data[task],
                imgsz,
                batch_size,
                stride,
                self.args.single_cls,
                pad=pad,
                rect=rect,
                workers=self.args.workers,
                prefix=colorstr(f"{task}: "),
            )[0]

        seen = 0
        confusion_matrix = ConfusionMatrix(nc=nc)
        names = model.names if hasattr(model, "names") else model.module.names  # get class names
        if isinstance(names, (list, tuple)):  # old format
            names = dict(enumerate(names))
        class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
        s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
        tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        dt = Profile(device=device), Profile(device=device), Profile(device=device)  # profiling times
        loss = torch.zeros(3, device=device)
        jdict, stats, ap, ap_class = [], [], [], []
        self.callbacks.run("on_val_start")
        pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            self.callbacks.run("on_val_batch_start")
            with dt[0]:
                if cuda:
                    im = im.to(device, non_blocking=True)
                    targets = targets.to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                nb, _, height, width = im.shape  # batch size, channels, height, width

            # Inference
            with dt[1]:
                preds, train_out = model(im) if self.args.compute_loss else (model(im, augment=self.args.augment), None)

            # Loss
            if self.args.compute_loss:
                loss += self.args.compute_loss(train_out, targets)[1]  # box, obj, cls

            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling
            with dt[2]:
                preds = non_max_suppression(
                    preds, self.args.conf_thres, self.args.iou_thres, labels=lb, multi_label=True, agnostic=self.args.single_cls, max_det=self.args.max_det
                )

            # Metrics
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                path, shape = Path(paths[si]), shapes[si][0]
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
                seen += 1

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                        if plots:
                            confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                    continue

                # Predictions
                if self.args.single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = self._process_batch(predn, labelsn, iouv)
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

                # Save/log
                if self.args.save_txt:
                    (save_dir / "labels").mkdir(parents=True, exist_ok=True)
                    self.save_one_txt(predn, self.args.save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
                if self.args.save_json:
                    self.save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
                self.callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

            # Plot images
            if plots and batch_i < 3:
                plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels
                plot_images(im, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred

            self.callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)

        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

        # Print results
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
        LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
        if nt.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

        # Print results per class
        if (self.args.verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        if not training:
            shape = (batch_size, 3, imgsz, imgsz)
            LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

        # Plots
        if plots:
            confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
            self.callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

        # Save JSON
        if self.args.save_json and len(jdict):
            w = Path(self.args.weights[0] if isinstance(self.args.weights, list) else self.args.weights).stem if self.args.weights is not None else ""  # weights
            anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations
            if not os.path.exists(anno_json):
                anno_json = os.path.join(data["path"], "annotations", "instances_val2017.json")
            pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
            with open(pred_json, "w") as f:
                json.dump(jdict, f)

            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                anno = COCO(anno_json)  # init annotations api
                pred = anno.loadRes(pred_json)  # init predictions api
                eval = COCOeval(anno, pred, "bbox")
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            except Exception as e:
                LOGGER.info(f"pycocotools unable to run: {e}")

        # Return results
        model.float()  # for training
        if not training:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
        
class DetectionPredictor_SAHI:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.save_dir = None
        self.device = None
        self.imgsz = None

def compile_validator(args, pt_modelpath, yaml_datapath, save_dir, imgsz = None, sahi = False, usual_inference=True):
    args.model = pt_modelpath
    args.data = yaml_datapath
    args.imgsz = imgsz

    validator = DetectionValidator_SAHI(args=args,save_dir=save_dir) if sahi else DetectionValidator(args=args,save_dir=save_dir)
    validator.is_coco = False
    validator.training = False

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=validator.args.model, force_reload=True)

    # Apply half precision if requested
    model.half() if validator.args.half else model.float()

    # Set validator parameters
    validator.names = model.names
    validator.nc = len(model.names)

    # Load dataset configuration
    validator.data = check_dataset(validator.args.data)

    # Inference modes
    validator.sahi = sahi
    validator.usual_inference = usual_inference

    # Set metrics
    validator.metrics.names = validator.names
    validator.metrics.plot = validator.args.plots

    validator.stride = model.stride
    LOGGER.info(f'\nValidator {"SAHI" if sahi else ""} compiled successfully!')
    return validator

def compile_predictor(args, pt_modelpath, save_dir, iou_thr = 0.5, conf = 0.5, imgsz = None, sahi = False, usual_inference = True):
    args.iou = iou_thr
    args.conf = conf
    args.model = pt_modelpath
    args.sahi = sahi
    args.usual_inference = usual_inference
    args.sahi_imgsz = imgsz
    args.dynamic_input = True if imgsz is None else False

    predictor = DetectionPredictor_SAHI(args = args) if sahi else DetectionPredictor()
    # predictor.model = pt_modelpath
    predictor.save_dir = save_dir
    # predictor.device = device
    # predictor.imgsz = imgsz
    # predictor.args.imgsz

    return predictor

def sahi_predict(detection_model, image_batch, slice_height = SLICE_H, slice_width = SLICE_W, \
                 overlap_height_ratio = OVERLAP_HEIGHT_RATIO, overlap_width_ratio = OVERLAP_WIDTH_RATIO):
    """
    detection_model : compiled from def get_sahi_model()
    image_batch : torch.Size([10, 3, 1280, 1280])
    """
    device_orig = detection_model.model.device
    batch_result = []
    for image in image_batch:
        box_annot = np.empty((0, 6)) #
        if isinstance(image,  torch.Tensor):
            image = image.cpu().numpy() * 255
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8) # in CHW

        result = get_sliced_prediction(
            image,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            verbose = VERBOSE_SAHI
        )
        for img_box in result.object_prediction_list:
            x1, y1, x2, y2 = img_box.bbox.to_xyxy()
            confidence = img_box.score.value
            cls = img_box.category.id
            box_annot = np.concatenate((box_annot, [[x1, y1, x2, y2, confidence, cls]]))
        batch_result.append(torch.tensor(box_annot).to(device_orig))
    return batch_result
