from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger
import os, cv2, numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances

register_coco_instances("painitngs", {}, "", "")
def detect_paintings(im, show=False):
    setup_logger()

    painting_metadata = MetadataCatalog.get("painitngs")
    painting_metadata.thing_classes = ["painitngs"]
    painting_metadata.thing_colors = [(0, 255, 0)]
    cfg = get_cfg()
    #cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    #os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join("detectron2_weights/detectron2_paintings_best.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode

    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=painting_metadata,
                   scale=1,
                   instance_mode=ColorMode.SEGMENTATION
                   )
    predictions = outputs["instances"].to("cpu")
    # masks = None
    #
    #
    masks = []
    paintings = []
    boxes = []

    for b in predictions.pred_boxes:
        box = b.numpy().astype("int")
        y1 = box[1] - 50 if box[1] - 50 > 0 else box[1]
        y2 = box[3] + 50 if box[3] + 50 < im.shape[0] else box[3]
        x1 = box[0] - 50 if box[0] - 50 > 0 else box[0]
        x2 = box[2] + 50 if box[2] + 50 < im.shape[1] else box[2]

        paintings.append(im[y1:y2, x1:x2])
        boxes.append(box)

    for m in predictions.pred_masks:
        mask = m.numpy().astype("uint8") * 255
        masks.append(mask)

    out = v.draw_instance_predictions(predictions)
    if show:
        cv2.imshow('pred_mask', out.get_image()[:, :, ::-1])
        key = cv2.waitKey(0)
    return np.array(out.get_image()[:, :, ::-1]), zip(paintings, boxes, masks)