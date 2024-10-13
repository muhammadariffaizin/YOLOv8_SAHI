import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

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
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode

from utils.metrics import box_iou
from utils.callbacks import Callbacks
from validator import BaseValidator

class DetectionValidator(BaseValidator):
    def __init__(self, args=None, save_dir=Path(""), model=None, dataloader=None):
        self.args = args
        self.callbacks = Callbacks()
        self.model = model
        self.dataloader = dataloader
        self.save_dir = save_dir

        self._validate_options()

    def _validate_options(self):
        """
        Validates and adjusts the configuration options.
        """
        if self.args.conf_thres > 0.001:
            LOGGER.warning(f"WARNING ⚠️ confidence threshold {self.args.conf_thres} > 0.001 produces invalid results")
        if self.args.save_hybrid:
            LOGGER.warning("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        self.args.save_json |= self.args.data.endswith("coco.yaml")
        self.args.save_txt |= self.args.save_hybrid

    def run_task(self):
        """
        Run the appropriate task ('train', 'val', 'test', 'speed', 'study') based on the configured options.
        """
        if self.args.task in ("train", "val", "test"):
            self._run_validation()

        elif self.args.task == "speed":
            self._run_speed_benchmarks()

        elif self.args.task == "study":
            self._run_study_benchmarks()

        else:
            raise NotImplementedError(f'--task {self.task} not in ("train", "val", "test", "speed", "study")')

    def _run_validation(self):
        """
        Perform validation tasks based on the options.
        """
        LOGGER.info(f"Running validation on task: {self.args.task}")
        # Here you would add the actual logic for training, validation, or testing
        self.run_model()

    def _run_speed_benchmarks(self):
        """
        Perform speed benchmarks.
        """
        self.args.conf_thres, self.args.iou_thres, self.args.save_json = 0.25, 0.45, False
        for weight in self.weights:
            self.args.weights = [weight]
            LOGGER.info(f"Running speed benchmark for weight: {weight}")
            self.run_model(plots=False)

    def _run_study_benchmarks(self):
        """
        Perform speed vs mAP study benchmarks.
        """
        for weight in self.args.weights:
            f = f"study_{Path(self.args.data).stem}_{Path(weight).stem}.txt"  # filename to save to
            x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
            for imgsz in x:  # img-size
                LOGGER.info(f"Running {f} --imgsz {imgsz}...")
                result, _, time_taken = self.run_model(plots=False)
                y.append(result + time_taken)  # results and times
            np.savetxt(f, y, fmt="%10.4g")  # save
        subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
        self.plot_val_study(x=x)

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

    def plot_val_study(self, x):
        """
        Plot validation study results.

        Args:
            x (list): List of x-axis values (e.g., image sizes).
        """
        # Here you would add code to plot the study results
        LOGGER.info(f"Plotting study results for {x}")

    def save_one_txt(self, predn, save_conf, shape, file):
        """
        Saves one detection result to a txt file in normalized xywh format, optionally including confidence.

        Args:
            predn (torch.Tensor): Predicted bounding boxes and associated confidence scores and classes in xyxy format, tensor
                of shape (N, 6) where N is the number of detections.
            save_conf (bool): If True, saves the confidence scores along with the bounding box coordinates.
            shape (tuple): Shape of the original image as (height, width).
            file (str | Path): File path where the result will be saved.

        Returns:
            None

        Notes:
            The xyxy bounding box format represents the coordinates (xmin, ymin, xmax, ymax).
            The xywh format represents the coordinates (center_x, center_y, width, height) and is normalized by the width and
            height of the image.

        Example:
            ```python
            predn = torch.tensor([[10, 20, 30, 40, 0.9, 1]])  # example prediction
            save_one_txt(predn, save_conf=True, shape=(640, 480), file="output.txt")
            ```
        """
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def save_one_json(self, predn, jdict, path, class_map):
        """
        Saves a single JSON detection result, including image ID, category ID, bounding box, and confidence score.

        Args:
            predn (torch.Tensor): Predicted detections in xyxy format with shape (n, 6) where n is the number of detections.
                                The tensor should contain [x_min, y_min, x_max, y_max, confidence, class_id] for each detection.
            jdict (list[dict]): List to collect JSON formatted detection results.
            path (pathlib.Path): Path object of the image file, used to extract image_id.
            class_map (dict[int, int]): Mapping from model class indices to dataset-specific category IDs.

        Returns:
            None: Appends detection results as dictionaries to `jdict` list in-place.

        Example:
            ```python
            predn = torch.tensor([[100, 50, 200, 150, 0.9, 0], [50, 30, 100, 80, 0.8, 1]])
            jdict = []
            path = Path("42.jpg")
            class_map = {0: 18, 1: 19}
            save_one_json(predn, jdict, path, class_map)
            ```
            This will append to `jdict`:
            ```
            [
                {'image_id': 42, 'category_id': 18, 'bbox': [125.0, 75.0, 100.0, 100.0], 'score': 0.9},
                {'image_id': 42, 'category_id': 19, 'bbox': [75.0, 55.0, 50.0, 50.0], 'score': 0.8}
            ]
            ```

        Notes:
            The `bbox` values are formatted as [x, y, width, height], where x and y represent the top-left corner of the box.
        """
        image_id = int(path.stem) if path.stem.isnumeric() else path.stem
        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            jdict.append(
                {
                    "image_id": image_id,
                    "category_id": class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def _process_batch(self, detections, labels, iouv):
        """
        Return a correct prediction matrix given detections and labels at various IoU thresholds.

        Args:
            detections (np.ndarray): Array of shape (N, 6) where each row corresponds to a detection with format
                [x1, y1, x2, y2, conf, class].
            labels (np.ndarray): Array of shape (M, 5) where each row corresponds to a ground truth label with format
                [class, x1, y1, x2, y2].
            iouv (np.ndarray): Array of IoU thresholds to evaluate at.

        Returns:
            correct (np.ndarray): A binary array of shape (N, len(iouv)) indicating whether each detection is a true positive
                for each IoU threshold. There are 10 IoU levels used in the evaluation.

        Example:
            ```python
            detections = np.array([[50, 50, 200, 200, 0.9, 1], [30, 30, 150, 150, 0.7, 0]])
            labels = np.array([[1, 50, 50, 200, 200]])
            iouv = np.linspace(0.5, 0.95, 10)
            correct = process_batch(detections, labels, iouv)
            ```

        Notes:
            - This function is used as part of the evaluation pipeline for object detection models.
            - IoU (Intersection over Union) is a common evaluation metric for object detection performance.
        """
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

class DetectionPredictor():
    def __init__(self):
        pass
