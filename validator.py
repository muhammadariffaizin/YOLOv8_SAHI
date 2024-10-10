import argparse
import torch
import numpy as np
import subprocess
from pathlib import Path
from utils.general import (
    LOGGER,
    xyxy2xywh,
)

from utils.metrics import box_iou
from utils.callbacks import Callbacks

class BaseValidator:
    def __init__(self, args=None, data="data/coco128.yaml", model=None, dataloader=None):
        self.args = args
        self.callbacks = Callbacks()
        self.model = model
        self.dataloader = dataloader
        self.save_dir = Path("")

        self._validate_options()

    def _validate_options(self):
        """
        Validates and adjusts the configuration options.
        """
        if self.conf_thres > 0.001:
            LOGGER.warning(f"WARNING ⚠️ confidence threshold {self.conf_thres} > 0.001 produces invalid results")
        if self.save_hybrid:
            LOGGER.warning("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        self.save_json |= self.data.endswith("coco.yaml")
        self.save_txt |= self.save_hybrid

    def run_task(self):
        """
        Run the appropriate task ('train', 'val', 'test', 'speed', 'study') based on the configured options.
        """
        if self.task in ("train", "val", "test"):
            self._run_validation()

        elif self.task == "speed":
            self._run_speed_benchmarks()

        elif self.task == "study":
            self._run_study_benchmarks()

        else:
            raise NotImplementedError(f'--task {self.task} not in ("train", "val", "test", "speed", "study")')

    def _run_validation(self):
        """
        Perform validation tasks based on the options.
        """
        LOGGER.info(f"Running validation on task: {self.task}")
        # Here you would add the actual logic for training, validation, or testing
        self.run_model()

    def _run_speed_benchmarks(self):
        """
        Perform speed benchmarks.
        """
        self.conf_thres, self.iou_thres, self.save_json = 0.25, 0.45, False
        for weight in self.weights:
            self.weights = [weight]
            LOGGER.info(f"Running speed benchmark for weight: {weight}")
            self.run_model(plots=False)

    def _run_study_benchmarks(self):
        """
        Perform speed vs mAP study benchmarks.
        """
        for weight in self.weights:
            f = f"study_{Path(self.data).stem}_{Path(weight).stem}.txt"  # filename to save to
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
        for k, v in self.opt.items():
            LOGGER.info(f"{k}: {v}")
        return "result", "other_data", "time_taken"

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

# Example of usage
validator = BaseValidator(
    data="data/coco128.yaml", weights=["yolov5s.pt", "yolov5m.pt"],
    task="val", batch_size=16, imgsz=640
)
validator.run_task()
