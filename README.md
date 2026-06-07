# Week 4: Object Detection and Segmentation Model

**Course:** Artificial Intelligence in Computer Vision  
**Dataset:** Pascal VOC 2012  
**Models:** YOLOv8 (object detection) + U-Net (segmentation)

---

## Objective

Build a combined object detection and segmentation system that identifies objects in images and outlines their precise boundaries. The scenario is a smart AI surveillance system that can not only detect what objects are present but also segment exactly where each object is in the frame.

---

## Scenario

Imagine you are building an AI-powered surveillance system for a security company. The system needs to:
1. Quickly detect all objects in a camera frame (people, vehicles, animals, etc.)
2. Precisely outline the boundaries of each detected object
3. Distinguish overlapping objects from each other

This is exactly what combining YOLO and U-Net achieves.

---

## Dataset Overview

**Pascal VOC 2012** (Visual Object Classes) is a standard computer vision benchmark dataset.

- **20 object categories:** aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, TV monitor
- Each image has a corresponding **segmentation mask** where each pixel is labeled with its object class
- Training set: ~1,464 images (we use a 200-image subset for speed)
- Image size used: 128×128 pixels

---

## Model Overview

### YOLO (Object Detection)

YOLO (You Only Look Once) is a real-time object detection model. We use the pre-trained **YOLOv8n** (nano) model from Ultralytics.

- Scans the full image in a single forward pass
- Returns bounding boxes, class labels, and confidence scores
- Pre-trained on COCO (80 classes); many overlap with VOC classes

### U-Net (Semantic Segmentation)

U-Net is a convolutional neural network designed specifically for image segmentation.

- **Encoder:** progressively reduces spatial resolution, extracting features
- **Decoder:** progressively upsamples back to original size
- **Skip connections:** pass encoder features directly to decoder to preserve fine detail
- Output: a class label for every single pixel
- Trained from scratch on Pascal VOC 2012 segmentation masks

### Combined Workflow

```
Input Image
    │
    ▼
[YOLO Detection] → bounding boxes + class labels
    │
    ▼
Crop detected region
    │
    ▼
[U-Net Segmentation] → pixel-wise class mask
    │
    ▼
Overlay mask on original image → final result
```

---

## Steps to Run the Project

1. Open `week4_object_detection_segmentation.ipynb` in VS Code or **Google Colab**
2. Run **Step 1** to install dependencies
3. Run **Step 2** to download and preprocess Pascal VOC 2012 (~2 GB download, one-time only — saved to `~/voc_data/`)
4. Run **Steps 3–4** to define and train U-Net (~15–20 minutes on CPU, ~5 minutes on GPU)
5. Run **Steps 5–6** to load YOLO and run detections
6. Run **Step 7** to execute the combined pipeline
7. Review all saved images in the `outputs/` folder

**Note:** The notebook is configured to run on CPU or GPU automatically. If using Google Colab, go to Runtime → Change runtime type → GPU for faster training.

---

## Required Libraries

```
ultralytics
torch
torchvision
opencv-python
numpy
matplotlib
pillow
```

Install with:
```bash
pip install ultralytics torch torchvision opencv-python numpy matplotlib pillow
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/dataset_samples.png` | Sample VOC images and ground truth masks |
| `outputs/training_loss.png` | U-Net loss curve across epochs |
| `outputs/unet_predictions.png` | U-Net predictions vs. ground truth |
| `outputs/yolo_detections_grid.png` | YOLO bounding box results on 4 images |
| `outputs/yolo_detection_*.png` | Individual YOLO detection images |
| `outputs/original_image.png` | Original input image |
| `outputs/yolo_output.png` | YOLO annotated image |
| `outputs/unet_mask.png` | U-Net predicted segmentation mask |
| `outputs/combined_result.png` | Final detection + segmentation overlay |
| `outputs/combined_pipeline_*.png` | Full 5-panel pipeline visualization |
| `outputs/unet_weights.pth` | Saved U-Net model weights |

---

## Results Summary

- **U-Net** trained for 5 epochs on 200 images; loss decreased consistently across all epochs (2.38 → 1.62 → 1.30 → 1.17 → 1.14), confirming the model is learning to segment Pascal VOC classes
- **YOLO** successfully detected objects (primarily people) with confidence scores up to 77% on Pascal VOC images using pre-trained YOLOv8n weights
- **Combined pipeline** successfully crops YOLO-detected regions, passes them through U-Net, and overlays segmentation masks on the original image
- Since U-Net was trained for only 5 epochs on 200 images, predictions default toward the background class due to pixel-level class imbalance; more training data and epochs would produce meaningful per-class segmentation

---

## Limitations

- U-Net is trained on only 200 images for 5 epochs — predictions default toward the background class due to pixel-level class imbalance, which is expected at this scale; real segmentation quality requires substantially more data and training time
- YOLO is pre-trained on COCO, not VOC specifically; some VOC classes may not be detected as precisely
- Image size is reduced to 128×128 for speed, which limits fine-grained detail
- The combined pipeline applies segmentation only to the highest-confidence YOLO detection

---

## References

Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The Pascal Visual Object Classes (VOC) challenge. *International Journal of Computer Vision, 88*(2), 303–338. https://doi.org/10.1007/s11263-009-0275-4

GeeksforGeeks. (2024, August 13). *Detect an object with OpenCV Python*. GeeksforGeeks. https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (Version 8.0.0) [Software]. https://github.com/ultralytics/ultralytics

Lakshmanan, V., Görner, M., & Gillard, R. (n.d.). *Practical machine learning for computer vision*. O'Reilly Media. https://learning.oreilly.com/library/view/practical-machine-learning/9781098102357/ch04.html

PyTorch. (n.d.). *torch.nn — PyTorch documentation*. https://pytorch.org/docs/stable/nn.html

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In N. Navab, J. Hornegger, W. M. Wells, & A. F. Frangi (Eds.), *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015* (pp. 234–241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28

Torchvision. (n.d.). *torchvision.datasets.VOCSegmentation*. PyTorch. https://pytorch.org/vision/stable/generated/torchvision.datasets.VOCSegmentation.html
