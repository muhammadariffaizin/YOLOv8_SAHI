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
from predictor import BasePredictor

class DetectionValidator(BaseValidator):
    def __init__(self, data="data/coco128.yaml", weights="yolov5s.pt", batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6,
                 max_det=300, task="val", device="", workers=8, single_cls=False, augment=False, verbose=False,
                 save_txt=False, save_hybrid=False, save_conf=False, save_json=False, project="runs/val",
                 name="exp", exist_ok=False, half=False, dnn=False, model=None, dataloader=None, save_dir=Path('')):
        super().__init__(data, weights, batch_size, imgsz, conf_thres, iou_thres, max_det, task, device, workers, single_cls,
                         augment, verbose, save_txt, save_hybrid, save_conf, save_json, project, name, exist_ok, half, dnn)
        self.model = model
        self.dataloader = dataloader
        self.save_dir = save_dir

    def __call__(self):
        return self.run(
            data=self.data,
            weights=self.weights,
            batch_size=self.batch_size,
            imgsz=self.imgsz,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=self.max_det,
            task=self.task,
            device=self.device,
            workers=self.workers,
            single_cls=self.single_cls,
            augment=self.augment,
            verbose=self.verbose,
            save_txt=self.save_txt,
            save_hybrid=self.save_hybrid,
            save_conf=self.save_conf,
            save_json=self.save_json,
            project=self.project,
            name=self.name,
            exist_ok=self.exist_ok,
            half=self.half,
            dnn=self.dnn,
            model=self.model,
            dataloader=self.dataloader,
            save_dir=self.save_dir,
        )
    
    @smart_inference_mode()
    def run(
        self,
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task="val",  # train, val, test, speed or study
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / "runs/val",  # save to project/name
        name="exp",  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(""),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
    ):
        """
        Evaluates a YOLOv5 model on a dataset and logs performance metrics.

        Args:
            data (str | dict): Path to a dataset YAML file or a dataset dictionary.
            weights (str | list[str], optional): Path to the model weights file(s). Supports various formats including PyTorch,
                TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TensorFlow SavedModel, TensorFlow GraphDef, TensorFlow Lite,
                TensorFlow Edge TPU, and PaddlePaddle.
            batch_size (int, optional): Batch size for inference. Default is 32.
            imgsz (int, optional): Input image size (pixels). Default is 640.
            conf_thres (float, optional): Confidence threshold for object detection. Default is 0.001.
            iou_thres (float, optional): IoU threshold for Non-Maximum Suppression (NMS). Default is 0.6.
            max_det (int, optional): Maximum number of detections per image. Default is 300.
            task (str, optional): Task type - 'train', 'val', 'test', 'speed', or 'study'. Default is 'val'.
            device (str, optional): Device to use for computation, e.g., '0' or '0,1,2,3' for CUDA or 'cpu' for CPU. Default is ''.
            workers (int, optional): Number of dataloader workers. Default is 8.
            single_cls (bool, optional): Treat dataset as a single class. Default is False.
            augment (bool, optional): Enable augmented inference. Default is False.
            verbose (bool, optional): Enable verbose output. Default is False.
            save_txt (bool, optional): Save results to *.txt files. Default is False.
            save_hybrid (bool, optional): Save label and prediction hybrid results to *.txt files. Default is False.
            save_conf (bool, optional): Save confidences in --save-txt labels. Default is False.
            save_json (bool, optional): Save a COCO-JSON results file. Default is False.
            project (str | Path, optional): Directory to save results. Default is ROOT/'runs/val'.
            name (str, optional): Name of the run. Default is 'exp'.
            exist_ok (bool, optional): Overwrite existing project/name without incrementing. Default is False.
            half (bool, optional): Use FP16 half-precision inference. Default is True.
            dnn (bool, optional): Use OpenCV DNN for ONNX inference. Default is False.
            model (torch.nn.Module, optional): Model object for training. Default is None.
            dataloader (torch.utils.data.DataLoader, optional): Dataloader object. Default is None.
            save_dir (Path, optional): Directory to save results. Default is Path('').
            plots (bool, optional): Plot validation images and metrics. Default is True.
            callbacks (utils.callbacks.Callbacks, optional): Callbacks for logging and monitoring. Default is Callbacks().
            compute_loss (function, optional): Loss function for training. Default is None.

        Returns:
            dict: Contains performance metrics including precision, recall, mAP50, and mAP50-95.
        """
        # Initialize/load model and set device
        training = model is not None
        if training:  # called by train.py
            device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
            half &= device.type != "cpu"  # half precision only supported on CUDA
            model.half() if half else model.float()
        else:  # called directly
            device = select_device(device, batch_size=batch_size)

            # Directories
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Load model
            model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_img_size(imgsz, s=stride)  # check image size
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
        nc = 1 if single_cls else int(data["nc"])  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        # Dataloader
        if not training:
            if pt and not single_cls:  # check --weights are trained on --data
                ncm = model.model.nc
                assert ncm == nc, (
                    f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
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
                single_cls,
                pad=pad,
                rect=rect,
                workers=workers,
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
        callbacks.run("on_val_start")
        pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            callbacks.run("on_val_batch_start")
            with dt[0]:
                if cuda:
                    im = im.to(device, non_blocking=True)
                    targets = targets.to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                nb, _, height, width = im.shape  # batch size, channels, height, width

            # Inference
            with dt[1]:
                preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

            # Loss
            if compute_loss:
                loss += compute_loss(train_out, targets)[1]  # box, obj, cls

            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            with dt[2]:
                preds = non_max_suppression(
                    preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
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
                if single_cls:
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
                if save_txt:
                    (save_dir / "labels").mkdir(parents=True, exist_ok=True)
                    self.save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
                if save_json:
                    self.save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
                callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

            # Plot images
            if plots and batch_i < 3:
                plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels
                plot_images(im, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred

            callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)

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
        if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
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
            callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

        # Save JSON
        if save_json and len(jdict):
            w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
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
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
    
class DetectionPredictor(BasePredictor):
    def __init__():
        pass