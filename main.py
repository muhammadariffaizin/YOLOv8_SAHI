import warnings
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from validation_sahi import run_sahi_validation, run_basic_validation, LOGGER
from inference_sahi import run_sahi_prediction, run_basic_prediction
import torch

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Ultralytics YOLOv5')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-file', type=str, default='data/coco.yaml', help='hyp.yaml file path')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam

def main():
    warnings.filterwarnings("ignore")
    
    # parse arguments
    args = parse_args()
    weights = args.weights
    imgsz = args.imgsz
    yaml_datapath = args.conf_file
    predict_source = args.source

    # defaults params
    args = get_cfg(cfg=DEFAULT_CFG)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    LOGGER.info(f'DEVICE ===>> {device}')

    # VALIDATION
    print(run_basic_validation(weights, yaml_datapath, args, imgsz))
    print(run_sahi_validation(weights, yaml_datapath, args, imgsz))

    # PREDICTION (INFERENCE)
    run_sahi_prediction(args, weights, source = predict_source, imgsz = imgsz)
    run_basic_prediction(pt_model=weights, args = args, source=predict_source)

if __name__ == '__main__':
    main()