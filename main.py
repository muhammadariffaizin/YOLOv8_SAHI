import warnings
from config import get_cfg
from utils import DEFAULT_CFG
from validation_sahi import run_sahi_validation, run_basic_validation, LOGGER
from inference_sahi import run_sahi_prediction, run_basic_prediction
import torch

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Ultralytics YOLOv5')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, help='inference size (pixels)')
    parser.add_argument('--conf-file', type=str, default='data/coco.yaml', help='hyp.yaml file path')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--mode', type=str, default='predict', help='select mode for valid or predict', choices=['predict', 'valid'])

    return parser.parse_args()

def main():
    warnings.filterwarnings("ignore")
    
    # parse arguments
    args = parse_args()
    weights = args.weights
    if args.imgsz == None:
        imgsz = None
    else:
        imgsz = (args.imgsz, args.imgsz) # (height, width) tuple
    yaml_datapath = args.conf_file
    predict_source = args.source

    # defaults params
    if args.conf_file:
        config = get_cfg(cfg=args.conf_file)
    else:
        config = get_cfg(cfg=DEFAULT_CFG)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    LOGGER.info(f'DEVICE ===>> {device}')
    print(imgsz)

    if args.mode == 'valid':
        print(run_basic_validation(pt_model=weights, yaml_datapath=yaml_datapath, args=config, imgsz=imgsz))
        print(run_sahi_validation(pt_model=weights, yaml_datapath=yaml_datapath, args=config, imgsz=imgsz))
    elif args.mode == 'predict':
        run_basic_prediction(args=config, pt_model=weights, source=predict_source)
        run_sahi_prediction(args=config, pt_model=weights, source=predict_source, imgsz=imgsz)

if __name__ == '__main__':
    main()