# Professional Real-Time Object Detection and Multi-Object Tracking

This project provides a modular Python system for real-time object detection, multi-object tracking, event analytics, logging, screenshots, and automatic annotated video export.

Default stack:

- Detector: Ultralytics YOLO11 COCO pretrained models
- Tracker: ByteTrack with a tuned tracker YAML
- UI: Streamlit dashboard, plus an optional Flask streaming dashboard
- Runtime: OpenCV video pipeline with GPU acceleration when CUDA is available

The COCO model detects all 80 standard classes, including people, cars, bicycles, motorcycles, trucks, dogs, cats, backpacks, handbags, suitcases, bottles, cell phones, chairs, laptops, TVs, traffic lights, and more.

live model: https://codealphaobjectdetector-usce78iqglwyvgikg7fajf.streamlit.app/

## Project Structure

```text
.
├── app.py
├── detect.py
├── flask_app.py
├── config.py
├── requirements.txt
├── models/
│   └── detector.py
├── tracker/
│   ├── custom_bytetrack.yaml
│   └── tracking_pipeline.py
├── utils/
│   ├── analytics.py
│   ├── coco.py
│   ├── colors.py
│   ├── drawing.py
│   ├── events.py
│   ├── fps.py
│   └── video.py
├── ui/
│   ├── streamlit_ui.py
│   ├── styles.py
│   ├── static/
│   │   └── flask.css
│   └── templates/
│       └── flask_index.html
├── logs/
└── outputs/
```

## Installation

Use Python 3.10 or newer. For best performance, use an NVIDIA GPU with a CUDA-enabled PyTorch build.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch from the official PyTorch selector, then install the remaining requirements:

```bash
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Download YOLO Weights

Ultralytics downloads official model weights automatically the first time they are used:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
```

Recommended models:

- `yolo11s.pt`: faster, lower accuracy
- `yolo11m.pt`: balanced default
- `yolo11l.pt`: high accuracy
- `yolo11x.pt`: maximum COCO accuracy, slower

You can also use YOLOv8 weights such as `yolov8m.pt`.

## Run Streamlit App

```bash
streamlit run app.py
```

Use the sidebar to select webcam, uploaded video, or IP/RTSP source. The app shows bounding boxes, labels, confidence percentage, persistent tracking IDs, live object counts, FPS, line crossing events, restricted zone events, screenshots, logs, and output video path.

## Run Flask App

```bash
python flask_app.py
```

Open:

```text
http://localhost:5000
```

## Run CLI

Webcam:

```bash
python detect.py --source 0 --model yolo11m.pt --conf 0.35 --iou 0.55 --imgsz 960 --save-video
```

Video file:

```bash
python detect.py --source path\to\video.mp4 --model yolo11l.pt --conf 0.30 --iou 0.55 --imgsz 1280 --save-video
```

Headless processing:

```bash
python detect.py --source path\to\video.mp4 --no-display --save-video
```

## Accuracy Settings

Use these settings as starting points:

- General real-time webcam: `model=yolo11m.pt`, `conf=0.35`, `iou=0.55`, `imgsz=960`
- Higher accuracy: `model=yolo11l.pt` or `yolo11x.pt`, `conf=0.25-0.40`, `imgsz=1280`
- Fewer false positives: increase confidence to `0.45-0.60`
- Better small-object recall: increase image size to `1280`
- Crowded scenes: keep NMS IoU around `0.50-0.65`
- Low light: enable low-light enhancement and use `yolo11l.pt` or custom training
- Fast CPU mode: use `yolo11n.pt` or `yolo11s.pt`, `imgsz=640`

The tracker is tuned in `tracker/custom_bytetrack.yaml`:

- `track_high_thresh`: raises or lowers first-stage matching strictness
- `track_low_thresh`: allows recovery of low-confidence detections
- `track_buffer`: keeps lost tracks alive during short occlusions
- `match_thresh`: controls association strictness

## Outputs

- Detection history CSV: `logs/detection_history.csv`
- Screenshots: `outputs/screenshots/`
- Annotated videos: `outputs/`
- Uploaded videos: `outputs/uploads/`

## Train On Custom Datasets

COCO pretrained models are strong for common classes, but custom training is the best way to improve accuracy for your own cameras, lighting, viewpoints, object sizes, uniforms, product packaging, or domain-specific classes.

Dataset format:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

Example `data.yaml`:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: helmet
  2: vehicle
```

Train:

```bash
yolo detect train model=yolo11m.pt data=dataset/data.yaml epochs=100 imgsz=960 batch=8 device=0 patience=30
```

Validate:

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=dataset/data.yaml imgsz=960
```

Run your trained model:

```bash
streamlit run app.py
```

Then enter the custom model path in code or run the CLI:

```bash
python detect.py --source 0 --model runs/detect/train/weights/best.pt --save-video
```

## Professional Accuracy Checklist

- Use the largest YOLO11 model your hardware can run at the required FPS.
- Use good camera placement, enough illumination, and minimal motion blur.
- Tune confidence per scene instead of using one global value for every camera.
- Increase `imgsz` for small objects such as bottles and phones.
- Train a custom model when the target objects differ from COCO or appear in unusual lighting.
- Keep ByteTrack thresholds stricter for clean scenes and more permissive for occlusion-heavy scenes.
- Prefer GPU inference with FP16 enabled for real-time performance.

## References

- Ultralytics tracking docs: https://docs.ultralytics.com/modes/track/
- Ultralytics YOLO11 model docs: https://docs.ultralytics.com/models/yolo11/
- ByteTrack tracker configuration: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml

