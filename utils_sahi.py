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
import cv2
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms import v2

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

def save_image_with_predictions(image, predictions, save_path):
    image = image[0].permute(1, 2, 0).cpu().numpy() * 255
    predictions = predictions.cpu().numpy()
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for box in predictions:
        xyxy = box[2:]
        x1, y1, x2, y2 = xyxy.tolist()
        x1 = int(x1 * image.shape[1])
        x2 = int(x2 * image.shape[1])
        y1 = int(y1 * image.shape[0])
        y2 = int(y2 * image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(save_path, image)

class DetectionValidator_SAHI(BaseValidator):
    def __init__(self, args=None, save_dir=Path(""), model=None, dataloader=None):
        super().__init__(args, save_dir, model, dataloader)

        self.nt_per_class = None
        self.class_map = None
        
        self._validate_options()
        
    def compute_window_shape(self, orig_shape, window_shape=[SLICE_H, SLICE_W]):
        """
        - we need compile model in advance with appropriate image size (yolo modela require imgsize be multiple 32)
        - that function fix cases when inpur original resolution lower than required window-sahi size (for example 720 * 1080 == Yolo can't process 720 imgsize)
        :param orig_shape: (720, 1280) tuple
        :param window_shape: hyperparameter
        :return: new shape model can be compiled with
        """
        h_orig, w_orig = orig_shape
        if window_shape[0] > h_orig: window_shape[0] = (h_orig | 31) - 31
        if window_shape[1] > w_orig: window_shape[1] = (w_orig | 31) - 31
        return window_shape

    def sahi_inference(
            self,
            im,
            slice_height=SLICE_H,
            slice_width=SLICE_W,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_WIDTH_RATIO,
            *args,
            **kwargs,
    ):
        """
        detection_model : compiled from def get_sahi_model()
        image_batch : torch.Size([10, 3, 1280, 1280])
        run_usual : if run usual inference for all picture (not sliced) above sahi prediction
        """
        btch, ch, image_height, image_width = im.shape
        # 1 - get sliced image coordinates
        slice_bboxes = get_slice_bboxes(
            image_height=image_height,
            image_width=image_width,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        # 2 - in loop do usual inference for each slice -> get predictions for each slice, append it to result list
        # check later gpu availability
        preds_all_slice_shifted = []
        for x1, y1, x2, y2 in slice_bboxes:
            # take all slices as batch = im and do inference on batch
            preds = self.model(
                im[:, :, y1:y2, x1:x2], *args, **kwargs
            )  # im here is batch !!
            if isinstance(
                    preds, (list, tuple)
            ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
                preds_take = preds[0]
            else:
                preds_take = preds
            preds_t = preds_take.transpose(-1, -2)
            prediction_slice = preds_t[..., :4]  # in xywh
            # shift prediction regarding to slice coordinates. got pregictionn regarding scale of original image
            prediction_slice[:, :, 0] += x1  # shift x center
            prediction_slice[:, :, 1] += y1  # shift y center
            preds_t[..., :4] = prediction_slice  # shift preds
            if isinstance(
                    preds, (list, tuple)
            ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
                preds[0] = preds_t.transpose(-1, -2)
                preds_all_slice_shifted.append(preds[0])
            else:
                preds = preds_t.transpose(-1, -2)
                preds_all_slice_shifted.append(preds)
        preds_all_slice_shifted_t = torch.cat((preds_all_slice_shifted), dim=2)
        return preds_all_slice_shifted_t

    def run_model(self, plots=True, trainer=None):
        """
        Run the model with the current options.

        Args:
            plots (bool): Whether or not to generate plots.
        """
        # Here the actual model inference logic goes, for example:
        # model = torch.load(self.weights[0])
        # result = model(img)
        # LOGGER.info("Model running with the following options:")
        # for k, v in self.args.__dict__.items():
        #     LOGGER.info(f"{k}: {v}")

        # Initialize/load model and set device
        training = trainer is not None
        if training:  # called by train.py
            device, pt, jit, engine = next(trainer.parameters()).device, True, False, False  # get model device, PyTorch model
            half &= device.type != "cpu"  # half precision only supported on CUDA
            trainer.half() if half else trainer.float()
        else:  # called directly
            device = select_device(self.args.device, batch_size=self.args.batch_size)

            # Directories
            save_dir = increment_path(Path(self.args.project) / self.args.name, exist_ok=self.args.exist_ok)  # increment run
            (save_dir / "labels" if self.args.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Load model
            self.model = DetectMultiBackend(self.args.weights, device=device, dnn=self.args.dnn, data=self.args.data, fp16=self.args.half)
            stride, pt, jit, engine = self.model.stride, self.model.pt, self.model.jit, self.model.engine
            imgsz = check_img_size(self.args.imgsz, s=stride)  # check image size
            half = self.model.fp16  # FP16 supported on limited backends with CUDA
            if engine:
                self.batch_size = self.model.batch_size
            else:
                device = self.model.device
                if not (pt or jit):
                    self.batch_size = 1  # export.py models default to batch-size 1
                    LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")
                else:
                    self.batch_size = self.args.batch_size

            # Data
            data = check_dataset(self.data)  # check

        if self.args.imgsz is None:
            self.args.batch_size = 1
            LOGGER.info(f'Forcing batch=1 : self.args.imgsz is None\nOriginal image size will be used for SAHI-validation')

        # Configure
        self.model.eval()
        self.cuda = device.type != "cpu"
        self.is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
        self.model.half() if self.args.half else self.model.float()
        self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names
        if self.names:
            self.nc = len(self.names)
        elif self.args.single_cls:
            self.nc = 1
        else:
            self.nc = int(data["nc"])  # number of classes
        self.iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.stride = self.model.stride

        # Set metrics
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots

        if imgsz is not None:
            slice_h, slice_w = self.compute_window_shape((imgsz,imgsz), window_shape=[SLICE_H, SLICE_W])
            self.model.warmup(imgsz=(1 if pt else self.batch_size, 3, slice_h, slice_w))  # warmup

        # Dataloader
        if not training:
            if pt and not self.args.single_cls:  # check --weights are trained on --data
                ncm = self.model.model.nc
                assert ncm == self.nc, (
                    f"{self.args.weights} ({ncm} classes) trained on different --data than what you passed ({self.nc} "
                    f"classes). Pass correct combination of --weights and --data that are trained together."
                )
            self.model.warmup(imgsz=(1 if pt else self.batch_size, 3, imgsz, imgsz))  # warmup
            pad, rect = (0.0, False) if self.task == "speed" else (0.5, pt)  # square inference for benchmarks
            self.task = self.task if self.task in ("train", "val", "test") else "val"  # path to train/val/test images
            dataloader = create_dataloader(
                data[self.task],
                imgsz,
                self.batch_size,
                stride,
                self.args.single_cls,
                pad=pad,
                rect=rect,
                workers=self.args.workers,
                prefix=colorstr(f"{self.task}: "),
            )[0]

        seen = 0
        confusion_matrix = ConfusionMatrix(nc=self.nc)
        names = self.model.names if hasattr(self.model, "names") else self.model.module.names  # get class names
        if isinstance(names, (list, tuple)):  # old format
            names = dict(enumerate(names))
        class_map = coco80_to_coco91_class() if self.is_coco else list(range(1000))
        s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
        tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        dt = Profile(device=device), Profile(device=device), Profile(device=device)  # profiling times
        loss = torch.zeros(3, device=device)
        jdict, stats, ap, ap_class = [], [], [], []
        self.callbacks.run("on_val_start")
        pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar

        if self.args.imgsz is None:
            LOGGER.info(f'started monkeypatching')
            import types
            from validation_loader import load_image, get_image_and_label
            self.dataloader.dataset.load_image = types.MethodType(load_image, self.dataloader.dataset)
            self.dataloader.dataset.get_image_and_label = types.MethodType(get_image_and_label, self.dataloader.dataset)
            LOGGER.info(f'monkeypatching done')

        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            if self.args.imgsz is None: # must calculate window shape here foe original-sized prediction
                imgsz = im.shape[2:]
                slice_h, slice_w = self.compute_window_shape(imgsz, window_shape=[SLICE_H, SLICE_W])
                self.model.warmup(imgsz=(1 if pt else self.args.batch, 3, slice_h, slice_w))
                # slice_h, slice_w = self.compute_window_shape(self.imgsz, window_shape=[SLICE_H, SLICE_W])

            if self.args.plots and batch_i < 3:
                self.plot_val_samples((im, targets, paths, shapes), batch_i)

            self.callbacks.run("on_val_batch_start")
            with dt[0]:
                if self.cuda:
                    im = im.to(device, non_blocking=True)
                    targets = targets.to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                nb, _, height, width = im.shape  # batch size, channels, height, width

                im_sahi = im.clone()
                im_sahi = v2.Resize(size=(slice_h, slice_w))(im_sahi) # image for not-sahi inference
                from_shape = im_sahi.shape[2:]
                to_shape = im.shape[2:]

                print("from shape ", from_shape)
                print("to shape ", to_shape)

            # Inference
            with dt[1]:
                if self.usual_inference:
                    preds, train_out = self.model(im_sahi) if self.args.compute_loss else (self.model(im_sahi, augment=self.args.augment), None)
                    w_gain, h_gain = (
                        to_shape[1] / from_shape[1],
                        to_shape[0] / from_shape[0],
                    )
                    if isinstance(preds, (list, tuple)):
                        transposed = preds[0]
                        for box in transposed[..., :4]:
                            box[:, 0] *= w_gain
                            box[:, 1] *= h_gain
                            box[:, 2] *= w_gain
                            box[:, 3] *= h_gain
                        scaled_xywh = transposed.transpose(-1, -2)
                        preds[0] = scaled_xywh
                    else:
                        transposed = preds
                        for box in transposed[..., :4]:
                            box[:, 0] *= w_gain
                            box[:, 1] *= h_gain
                            box[:, 2] *= w_gain
                            box[:, 3] *= h_gain
                        scaled_xywh = transposed.transpose(-1, -2)
                        preds = scaled_xywh
                if self.sahi:
                    preds_sahi = self.sahi_inference(im=im_sahi, slice_height=slice_h, slice_width=slice_w, augment=False)
                    preds_sahi = preds_sahi.transpose(-1, -2)
                if self.sahi and self.usual_inference:
                    if isinstance(preds, (list, tuple)):
                        preds[0] = torch.cat((preds[0], preds_sahi), dim=2)
                    else:
                        preds = torch.cat((preds, preds_sahi), dim=2)

            # Loss
            if self.args.compute_loss:
                loss += self.args.compute_loss(train_out, targets)[1]  # box, obj, cls

            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling
            with dt[2]:
                if not self.usual_inference:
                    preds = preds_sahi
                if isinstance(preds, (list, tuple)):
                    preds[0] = preds[0].transpose(-1, -2)
                else:
                    preds = preds.transpose(-1, -2)
                preds = non_max_suppression(
                    preds, self.args.conf_thres, self.args.iou_thres, labels=lb, multi_label=True, agnostic=self.args.single_cls, max_det=self.args.max_det
                )

            # Metrics
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                path, shape = Path(paths[si]), shapes[si][0]
                correct = torch.zeros(npr, self.niou, dtype=torch.bool, device=device)  # init
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
                scale_boxes(im_sahi[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_boxes(im_sahi[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = self._process_batch(predn, labelsn, self.iouv)
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
        nt = np.bincount(stats[3].astype(int), minlength=self.nc)  # number of targets per class

        # Print results
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
        LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
        if nt.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.task} set, can not compute metrics without labels")

        # Print results per class
        if (self.args.verbose or (self.nc < 50 and not training)) and self.nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        if not training:
            shape = (self.batch_size, 3, imgsz, imgsz)
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
                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            except Exception as e:
                LOGGER.info(f"pycocotools unable to run: {e}")

        # Return results
        self.model.float()  # for training
        if not training:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        maps = np.zeros(self.nc) + map
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
    args.weights = pt_modelpath
    args.data = yaml_datapath
    args.imgsz = imgsz

    if sahi:
        validator = DetectionValidator_SAHI(args=args, save_dir=save_dir, model=pt_modelpath)
    else:
        validator = DetectionValidator(args=args, save_dir=save_dir, model=pt_modelpath)
    validator.is_coco = False
    validator.training = False

    # Load dataset configuration
    validator.data = check_dataset(validator.args.data)

    # Inference modes
    validator.sahi = sahi
    validator.usual_inference = usual_inference

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
