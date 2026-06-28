# Bangladesh Traffic Perception System

## Detailed Thesis Summary and Research Progress Report

**Institution:** International University of Business Agriculture and Technology (IUBAT)  
**Department:** Computer Science and Engineering  
**Research Area:** Computer Vision, Autonomous Systems, Advanced Driver Assistance Systems (ADAS), and Intelligent Transportation Systems (ITS)  
**Project Ecosystem:** Bangladesh Traffic Perception System, developed as a core perception component for the broader **Omars' Valley** autonomous infrastructure concept  
**Authors:** Md. Taher Bin Omar Hijbullah and Md. Rony Mia  
**Primary Model Family:** YOLOv11 / YOLO26m  
**Custom Architectural Modules:** CBAM, DCFM, DCFM_v2, Transformer-style attention concepts, SAHI-Pro/P2 small-object head concept  
**Main Traffic Region:** Bangladesh, especially dense Dhaka-style urban traffic  

---

## Document Purpose

This document organizes our thesis work into a clear, detailed, and understandable research summary. It combines our updated experimental notes, architecture decisions, dataset engineering work, training results, deployment work, conference milestone, and future research direction.

The writing uses first-person plural wording such as **we**, **our**, and **us**, because this summary represents the work of our research team.

---

## Table of Contents

| Section | Topic |
| --- | --- |
| 1 | Research Overview |
| 2 | Problem Background |
| 3 | Research Objectives |
| 4 | End-to-End System Pipeline |
| 5 | Dataset Engineering |
| 6 | Experiment History |
| 7 | Architecture Modules |
| 8 | Result Comparison |
| 9 | Software, Files, and Artifacts |
| 10 | Deployment and Automation |
| 11 | Key Findings |
| 12 | Limitations |
| 13 | Current Status |
| 14 | Future Work |
| 15 | Final Summary |

---

## 1. Research Overview

Our thesis develops a YOLO-based object detection system for dense and unstructured Bangladeshi traffic scenes. The system is designed to detect local road users such as cars, buses, trucks, rickshaws, CNG/auto-rickshaws, bicycles/cycles, motorcycles, vans, pedestrians, and other common traffic objects.

Bangladeshi traffic is different from many structured road datasets. Road users often move without strict lane discipline, vehicles appear very close to each other, and local vehicle types such as rickshaws and CNGs are underrepresented in many global datasets. This makes standard traffic detection models less reliable in our local environment.

To address this challenge, we built an end-to-end research workflow that includes dataset preparation, annotation conversion, custom YOLO architecture design, attention module integration, multi-scale feature fusion, high-resolution training, SAHI analysis, Streamlit deployment, and future tracking/risk-assessment planning.

Our first major external milestone was the acceptance of our paper titled **"Attention-Guided YOLOv11 for Object Detection in Unstructured Bangladeshi Traffic"** at the **2026 International Conference on Engineering and Frontier Technologies (ICEFT 2026)** in Tangail, Bangladesh.

---

## 2. Problem Background

Traffic perception systems for autonomous vehicles and ADAS applications must detect surrounding objects accurately and consistently. In countries with structured lane-based traffic, detection models often perform well because road geometry, vehicle appearance, and object movement are relatively predictable.

Bangladeshi traffic presents a much harder perception problem.

| Challenge | Explanation |
| --- | --- |
| Dense traffic | Many vehicles and pedestrians appear in one frame with very little spacing. |
| Non-lane-based movement | Road users do not always follow fixed lanes, so lane-based assumptions become weak. |
| Heavy occlusion | Rickshaws, cars, buses, motorcycles, pedestrians, and vans frequently overlap. |
| Small objects | Cycles, distant pedestrians, and partially visible vehicles can occupy very few pixels. |
| Local vehicle types | Rickshaws, CNGs, easy bikes, vans, and local transport patterns are not well represented in many global datasets. |
| Class imbalance | Some classes have many examples, while rare classes such as cycles have very few samples. |
| Background confusion | Shops, trees, road dividers, signs, shadows, and crowds can create false positives. |
| Visual similarity | Some vehicles share similar colors, shapes, and textures, making separation difficult. |

This research is important because a detector trained only on clean or western-style traffic scenes may not generalize properly to Bangladesh. Our work contributes a region-aware detection pipeline for local traffic perception.

---

## 3. Research Objectives

The main objectives of our thesis are:

1. Build a YOLO-based traffic object detection pipeline for Bangladeshi road scenes.
2. Convert and validate Pascal VOC XML annotations into YOLO format.
3. Create a clean dataset structure suitable for training, validation, and testing.
4. Reduce data leakage by using a strict video-wise dataset split in later experiments.
5. Evaluate baseline YOLO models on local traffic data.
6. Improve feature attention using CBAM.
7. Improve multi-scale representation using DCFM and DCFM_v2.
8. Study high-resolution training for small-object detection.
9. Test SAHI-style slicing and document its limitations for our dataset.
10. Build a deployable demonstration interface using Streamlit.
11. Prepare the system for future tracking and traffic risk assessment.

---

## 4. End-to-End System Pipeline

Our complete research pipeline is shown below.

```text
Raw Bangladeshi / Dhaka Traffic Frames
        |
        v
Pascal VOC XML Annotation Collection
        |
        v
XML-to-YOLO Label Conversion
        |
        v
Bounding-Box Normalization
        |
        v
Dataset Cleaning and Sanity Checking
        |
        v
Train / Validation / Test Split
        |
        +--> Early 80/20 cloud split
        +--> Later 70/15/15 strict video-wise split
        |
        v
Baseline YOLO Training
        |
        +--> YOLO11n
        +--> YOLOv11m
        +--> YOLOv11m + CBAM
        +--> YOLOv11m + DCFM
        +--> YOLO26m Dhaka local framework
        |
        v
Advanced Experiments
        |
        +--> 640px training
        +--> 416px stability run
        +--> 1024px high-resolution training
        +--> SAHI patch-training experiment
        +--> SAHI-Pro / P2 small-object-head concept
        |
        v
Evaluation, Error Analysis, Web Deployment, and Tracking Roadmap
```

This pipeline reflects our actual research journey from raw traffic data to a deployable perception system.

---

## 5. Dataset Engineering

### 5.1 Dataset Versions

| Dataset Stage | Classes | Images / Files | Split | Purpose |
| --- | ---: | ---: | --- | --- |
| V1 baseline dataset | 4 | 23,564 valid image-label pairs | 16,494 train / 4,712 validation / 2,358 test | First proof-of-concept detector |
| V2 expanded dataset | 9 | 23,648 processed files | 80% train / 20% validation | Expanded local traffic classes |
| V5 cloud baseline dataset | 9 | 18,919 train / 4,729 validation | Cloud validation setup | Standard YOLOv11m comparison |
| V6 Dhaka local dataset | 8 | 28,159 valid images | 70% train / 15% validation / 15% test | Harder Dhaka-style benchmark |
| V6 annotation scale | 8 | 289,506 object instances | Same as V6 | Dense traffic object benchmark |

### 5.2 Target Classes

Across the project, our class configuration changed as the dataset matured. Early experiments used four populated classes, while later experiments used expanded 8-class or 9-class settings.

Common classes included:

- Car
- Bus
- Truck
- Motorcycle/Bike
- Bicycle/Cycle
- Rickshaw
- CNG/Auto-rickshaw
- Mini-truck or Van
- Pedestrian/People

The exact class count depended on the dataset version and YAML configuration used in that experiment.

### 5.3 Annotation Conversion

The original labels were stored in Pascal VOC XML format. YOLO requires normalized center-based annotations:

```text
class_id x_center y_center width height
```

All bounding-box values must be normalized between `0.0` and `1.0`.

```python
def normalize_pascal_xml_to_yolo(x_min, x_max, y_min, y_max, img_width, img_height):
    x_center = (x_min + x_max) / (2.0 * img_width)
    y_center = (y_min + y_max) / (2.0 * img_height)
    bbox_width = (x_max - x_min) / img_width
    bbox_height = (y_max - y_min) / img_height
    return float(x_center), float(y_center), float(bbox_width), float(bbox_height)
```

This conversion step was essential because incorrect bounding boxes can completely damage YOLO training.

### 5.4 Dataset Validation

Before training, we used validation logic to check:

- Missing image-label pairs.
- Invalid class IDs.
- Bounding boxes outside the normalized range.
- Corrupted annotation files.
- Incorrect dataset YAML paths.
- Class-name mismatches.

Example validation logic:

```python
if class_id >= len(class_names) or class_id < 0:
    raise ValueError(f"Invalid class ID detected: {class_id}")

if x_center > 1.0 or y_center > 1.0 or box_w > 1.0 or box_h > 1.0:
    raise ValueError("Bounding-box coordinates are outside the normalized range [0, 1].")
```

### 5.5 Split Strategy

We used two main split strategies.

| Split Type | Used In | Reason |
| --- | --- | --- |
| 80/20 split | Early Kaggle and cloud experiments | Faster development and baseline testing |
| 70/15/15 video-wise split | Dhaka V6 local experiments | Stronger evaluation with less frame leakage |

The video-wise split is especially important because frames from the same video often share the same background, lighting, camera angle, and object movement. If near-identical frames are placed in both training and testing sets, the model may appear stronger than it actually is.

---

## 6. Chronological Experiment History

### 6.1 V1: Baseline Light Model

| Item | Details |
| --- | --- |
| Objective | Build our first custom-scaled traffic detector |
| Platform | Kaggle Kernel |
| GPU | NVIDIA Tesla T4 |
| Dataset | 4 populated classes: car, bus, truck, rickshaw |
| Valid pairs | 23,564 |
| Image size | 640x640 |
| Batch size | 16 |
| Epochs | 50 |
| Runtime | Approximately 2.77 hours |

| Metric | Result |
| --- | ---: |
| Precision | 72.46% |
| Recall | 70.55% |
| mAP@50 | 76.88% |
| mAP@50-95 | 54.62% |

**Finding:** This model produced strong initial results, but it exposed weaknesses around background confusion and minority classes such as trucks.

### 6.2 V2: Class Expansion and Data Formatting

| Item | Details |
| --- | --- |
| Objective | Expand the dataset from 4 classes to 9 classes |
| Main task | Pascal VOC XML to YOLO conversion |
| Main script | `fix_labels.py` |
| Processed files | 23,648 |
| Parsing success | Approximately 99.8% |
| Split | 80% train / 20% validation |
| Archive | `bangladesh_yolo_ready.zip` |
| Archive size | Approximately 1,605.53 MB |

**Finding:** This stage created a reusable dataset conversion foundation for later experiments.

### 6.3 V3: YOLOv11m-CBAM Attention Model

| Item | Details |
| --- | --- |
| Objective | Add attention to improve feature selection in dense traffic scenes |
| Architecture | YOLOv11m + CBAM |
| CBAM placement | Deep P5 feature layer / Layer 23 before detection |
| Platform | Kaggle DDP |
| GPU | 2x NVIDIA Tesla T4 |
| Classes | 9 |
| Image size | 640x640 |
| Batch size | 32 |
| Epochs | 50 |
| Runtime | 5.584 hours |

| Metric | Result |
| --- | ---: |
| Precision | 67.10% |
| Recall | 72.00% |
| mAP@50 | 74.90% |
| mAP@50-95 | 52.00% |
| Car mAP@50 | 93.30% |
| Rickshaw mAP@50 | 90.40% |
| Cycle mAP@50 | 31.50% |

**Finding:** CBAM helped the model focus on important regions, but it did not fully solve the rare-class problem.

### 6.4 V4: Targeted CBAM Fine-Tuning

| Item | Details |
| --- | --- |
| Objective | Fine-tune attention features for weak classes |
| Platform | Kaggle multi-GPU |
| GPU | 2x Tesla T4 |
| Training | 30 specialized epochs |
| Log file | `resultss.csv` |

| Metric | Result |
| --- | ---: |
| Global mAP@50 | 75.13% |
| Mini-Truck mAP@50 | 70.00% |
| Truck mAP@50 | 50.00% |
| Cycle mAP@50 | 32.00% |

**Finding:** Fine-tuning improved some difficult classes, but cycle detection stayed weak due to data scarcity.

### 6.5 V5: Standard YOLOv11m Reference Baseline

| Item | Details |
| --- | --- |
| Objective | Establish a fair unmodified YOLOv11m baseline |
| Architecture | Standard YOLOv11m |
| Platform | Kaggle DDP |
| GPU | 2x Tesla T4 |
| Training images | 18,919 |
| Validation frames | 4,729 |
| Validation targets | 52,941 |
| Epochs | 50 |
| Runtime | 6.928 hours |

| Metric | Result |
| --- | ---: |
| Precision | 69.70% |
| Recall | 71.40% |
| mAP@50 | 75.10% |
| mAP@50-95 | 52.80% |

**Finding:** The standard YOLOv11m result became a stable comparison point for CBAM and DCFM.

### 6.6 V5.2: DCFM and DCFM_v2 Multi-Scale Fusion

| Item | Details |
| --- | --- |
| Objective | Improve detection across different object scales |
| Module | DCFM / DCFM_v2 |
| Configuration | `yolov11m-dcfm.yaml` |
| Stability run image size | 416px |
| Stability run batch | 8 |
| Device | Single GPU, `device=0` |
| Backup strategy | Kaggle dataset backup every 5 epochs |

| Metric | Result |
| --- | ---: |
| Global recall | Approximately 73.1% |
| mAP@50 | Approximately 74.8% |

**Finding:** DCFM improved recall, which is valuable for safety-focused traffic perception because missed detections can be dangerous.

### 6.7 ICEFT 2026 Conference Milestone

| Item | Details |
| --- | --- |
| Paper title | Attention-Guided YOLOv11 for Object Detection in Unstructured Bangladeshi Traffic |
| Conference | 2026 International Conference on Engineering and Frontier Technologies (ICEFT 2026) |
| Location | Tangail, Bangladesh |
| Presentation date | January 28, 2026 |
| Conference site | <https://icefront.mbstu.ac.bd/> |
| Document record | `Attention_Guided_YOLOv11_for_Object_Detection_in_Unstructured_Bangladeshi_Traffic_c_1.pdf` |

| Published / Reported Item | Value |
| --- | ---: |
| Standard YOLOv11m baseline | 75.1% mAP |
| Custom YOLOv11m-CBAM model | 75.7% mAP |
| Reported latency | 4.7 ms / frame |

Class-wise ICEFT 2026 results:

| Target Class | Instances | Precision | Recall | mAP@50 |
| --- | ---: | ---: | ---: | ---: |
| Car | 63,107 | 83.3% | 88.5% | 93.4% |
| CNG / Auto-Rickshaw | 27,823 | 83.3% | 84.1% | 91.0% |
| Rickshaw | 60,241 | 78.5% | 85.6% | 90.5% |
| Bus / Microbus | 51,996 | 78.9% | 84.2% | 89.2% |
| Cycle | 714 | 45.8% | 27.0% | 31.6% |
| Global aggregate | 265,698 | 66.8% | 72.7% | 75.7% |

**Finding:** The paper validated our attention-guided YOLOv11-CBAM direction through peer review.

### 6.8 V6.1: Dhaka Local 640px Baseline

| Item | Details |
| --- | --- |
| Objective | Establish a baseline on the harder Dhaka V6 dataset |
| Platform | Google Colab |
| GPU | Tesla T4 |
| Architecture | Intermediate YOLO26m base |
| Image size | 640px |
| Epochs | 50 |
| Runtime | Approximately 4.20 hours |

| Metric / Class | Result |
| --- | ---: |
| Final mAP@50 | 48.20% |
| Rickshaw mAP@50 | 15.1% |
| Pedestrian / People mAP@50 | 8.1% |
| Bicycle / Cycle mAP@50 | 6.1% |

**Finding:** Dhaka V6 was much harder than the earlier dataset because of stronger occlusion, denser scenes, and smaller road users.

### 6.9 V6.2 - V6.4: Local 1024px High-Resolution Marathon

| Item | Details |
| --- | --- |
| Objective | Recover small-object details using higher resolution |
| Image size | 1024px |
| Platform | Local workstation |
| GPU | NVIDIA RTX 4000, 8GB VRAM |
| Runtime | 21.58 hours / 77,707 seconds |
| Peak epoch | Epoch 7 |

Augmentation settings:

| Augmentation | Value | Purpose |
| --- | ---: | --- |
| Mosaic | 1.0 | Simulate dense intersection occlusion |
| Copy-Paste | 0.30 | Increase minority-class exposure |
| MixUp | 0.15 | Improve overlap robustness |
| HSV Hue/Saturation/Value | 0.015 / 0.7 / 0.4 | Simulate local weather and lighting variation |

| Metric | Result |
| --- | ---: |
| mAP@50 | 49.67% |
| Global recall | 64.01% |

**Finding:** High-resolution training preserved more detail for small and distant objects.

### 6.10 V6.5: High-Resolution Baseline Peak

| Item | Details |
| --- | --- |
| Objective | Define the strongest high-resolution Dhaka baseline |
| Architecture | YOLO26m base |
| Image size | 1024px |
| Platform | Local RTX 4000 workstation |
| Epochs | 50 |
| Runtime | 33.82 hours / 121,738 seconds |
| Peak mAP epoch | Epoch 14 |

| Metric | Result |
| --- | ---: |
| Peak global mAP@50 | 49.78% |
| Global box recall | 67.90% |
| Strict localization mAP@50-95 | 33.12% |

**Finding:** This became our strongest V6 baseline. Recall improved significantly, but mAP remained below 50% because of background confusion, extreme crowding, and class imbalance.

### 6.11 V6_SAHI: Patch-Training Failure

| Item | Details |
| --- | --- |
| Objective | Test SAHI-style patch training |
| Script | `sahi_slicer.py` |
| Patch size | 640x640 overlapping windows |
| Runtime | 15.48 hours / 55,718 seconds |

| Metric | Result |
| --- | ---: |
| Global mAP@50 | 0.74% |
| Global mAP@50-95 | 0.16% |
| Validation classification loss | 7.18 |

**Finding:** SAHI patch training damaged global scene context. Large vehicles were split across patch boundaries, and the model learned partial fragments instead of complete objects. Our conclusion is that SAHI is better suited as an inference-time technique for this dataset, not as the main training format.

### 6.12 Advanced Direction: CBAM + DCFM + Transformer + SAHI-Pro

| Notebook / Concept | Role |
| --- | --- |
| `YOLOv11_+_DCFM_+_Transformer_+_CBAM_V6_2ipynb.ipynb` | Explores combined CBAM, DCFM, and Transformer-style attention ideas |
| `Ultimate_Thesis.ipynb` | Records Colab setup, Drive mounting, custom Ultralytics hijacking, SAHI YAML generation, and diagnostic training |
| `yolo26m-sahi-pro.yaml` | Proposes a SAHI-Pro architecture with P2 small-object head concepts |
| `Dhaka_Traffic_SAHI.yaml` | Dataset configuration for SAHI-related experiments |

**Finding:** Our next research horizon is stronger small-object handling through P2 heads, context-aware attention, and tracking.

---

## 7. Architecture Modules

### 7.1 CBAM: Convolutional Block Attention Module

CBAM improves feature selection by applying attention in two stages:

1. **Channel attention:** Learns which feature channels are important.
2. **Spatial attention:** Learns where important objects or object parts are located.

This is useful in dense traffic scenes because the model must distinguish vehicles from cluttered backgrounds and overlapping road users.

### 7.2 DCFM: Dynamic Channel Fusion Module

DCFM improves multi-scale representation by using parallel convolution paths. The module can process small, medium, and larger receptive fields before fusing the results.

The goal is to help the model detect:

- Small cycles and pedestrians.
- Medium vehicles such as CNGs and rickshaws.
- Large vehicles such as buses and trucks.
- Partially occluded traffic objects.

### 7.3 DCFM_v2

DCFM_v2 was created as a safer and more memory-aware version of the fusion module for cloud training. It used dynamic layer construction and reduced channel pressure to avoid CUDA out-of-memory errors.

Important engineering decisions included:

- Single-GPU execution instead of unstable DDP.
- Smaller image size during stability testing.
- Backup callback every 5 epochs.
- Re-injection of the custom class into the Ultralytics runtime before loading checkpoints.

### 7.4 Transformer / Self-Attention Direction

Transformer-style attention was considered as a future extension to improve global context modeling. This is relevant because local convolution alone may struggle when traffic objects are heavily occluded or spread across complex scenes.

### 7.5 YOLO26m Local Framework

YOLO26m was used for later Dhaka V6 experiments. The goal was to build a stronger local baseline and test higher-resolution training on the RTX 4000 workstation.

### 7.6 SAHI-Pro and P2 Small-Object Head Concept

The SAHI-Pro direction focuses on small-object detection. The P2 head concept is useful because early feature maps preserve more spatial detail than deeper feature maps. This may improve detection of cycles, pedestrians, and distant vehicles.

---

## 8. Main Experimental Comparison

| Metric | V1 Baseline | V3 CBAM | V5 YOLOv11m | V6.1 Dhaka Base | V6.4 1024px | V6.5 High-Res Peak | V6 SAHI Patch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model | YOLO11n | YOLOv11m-CBAM | YOLOv11m | YOLO26m | YOLO26m | YOLO26m | YOLO26m + SAHI |
| Classes | 4 | 9 | 9 | 8 | 8 | 8 | 8 |
| Image size | 640 | 640 | 640 | 640 | 1024 | 1024 | 1024 sliced |
| Platform | Kaggle | Kaggle | Kaggle | Colab | Local | Local | Local |
| GPU | Tesla T4 | 2x Tesla T4 | 2x Tesla T4 | Tesla T4 | RTX 4000 | RTX 4000 | RTX 4000 |
| Runtime | 2.77 h | 5.58 h | 6.93 h | 4.20 h | 21.58 h | 33.82 h | 15.48 h |
| Precision | 72.46% | 67.10% | 69.70% | 44.95% | 46.22% | 36.39% | 2.55% |
| Recall | 70.55% | 72.00% | 71.40% | 53.57% | 50.81% | 67.90% | 4.26% |
| mAP@50 | 76.88% | 74.90% | 75.10% | 48.20% | 49.67% | 49.78% | 0.74% |
| mAP@50-95 | 54.62% | 52.00% | 52.80% | 31.55% | 30.79% | 33.12% | 0.16% |

---

## 9. Software, Files, and Artifacts

### 9.1 Main Repository Files

| File | Purpose |
| --- | --- |
| `summery.md` | Current organized thesis summary |
| `Thesis Final Teport.pdf` | Final thesis report PDF currently present in the workspace |

### 9.2 Important Scripts Recorded in Our Notes

| Script | Purpose |
| --- | --- |
| `fix_labels.py` | Converts Pascal VOC XML labels into YOLO labels |
| `dataset_sanity_check.py` | Validates annotation files and bounding-box values |
| `make_yaml.py` | Generates dataset YAML files |
| `fix_yamls.py` | Corrects YAML paths and class metadata |
| `marathon.py` | Runs long training jobs |
| `high_res_marathon.py` | Runs high-resolution 1024px experiments |
| `sahi_slicer.py` | Creates SAHI-style sliced patches |
| `setup_files.py` | Stages files and creates backup archives |
| `app.py` | Streamlit web application interface |
| `inject.py` | Custom module injection support |
| `model_vision_check.py` | Model verification and visual checking |
| `risk_tracker.py` | Future tracking/risk-analysis support |

### 9.3 Dataset and Configuration Files

| File | Purpose |
| --- | --- |
| `Dhaka_Traffic_Local.yaml` | Main local Dhaka dataset configuration |
| `Dhaka_Traffic_SAHI.yaml` | SAHI-related dataset configuration |
| `yolo26m-cbam.yaml` | YOLO26m with CBAM configuration |
| `yolo26m-dcfm.yaml` | YOLO26m with DCFM configuration |
| `yolo26m-combined.yaml` | Combined architecture configuration |
| `yolo26m-sahi-pro.yaml` | Proposed SAHI-Pro / P2-head architecture |
| `yolov11m-cbam.yaml` | YOLOv11m-CBAM architecture used for conference-stage work |
| `yolov11m-dcfm.yaml` | YOLOv11m-DCFM architecture |

### 9.4 Notebooks and Documents

| Artifact | Purpose |
| --- | --- |
| `Ultimate_Thesis.ipynb` | Colab setup, dataset extraction, SAHI-Pro diagnostics |
| `YOLOv11_+_DCFM_+_Transformer_+_CBAM_V6_2ipynb.ipynb` | Advanced architecture exploration |
| `yolov11-dcfm-transformer-cbam-v5_kaggle.ipynb` | YOLOv11m-CBAM Kaggle architecture and training |
| `yolov11-dcfm-transformer-cbam-v5-2_kaggle.ipynb` | DCFM_v2 cloud stability run |
| `Attention_Guided_YOLOv11_for_Object_Detection_in_Unstructured_Bangladeshi_Traffic_c_1.pdf` | ICEFT 2026 paper record |

### 9.5 Local Project Structure

```text
D:\Hijbullah\Thesis\
|
+-- Master_Dataset_Local\
|   +-- images\
|   |   +-- train\
|   |   +-- val\
|   |   +-- test\
|   +-- labels\
|       +-- train\
|       +-- val\
|       +-- test\
|
+-- Ablation_Runs\
|   +-- Dhaka_CBAM_Only\
|   +-- Dhaka_DCFM_Only\
|   +-- Dhaka_Combined\
|
+-- High_Res_Runs\
|   +-- Dhaka_Combined_1024px\
|
+-- SAHI_Dataset\
|   +-- images\train\
|   +-- labels\train\
|
+-- Qualitative_Analysis\
+-- Git_Staging_Master\
+-- Dhaka_Traffic_Local.yaml
+-- Dhaka_Traffic_SAHI.yaml
```

---

## 10. Deployment and Automation

### 10.1 Streamlit Web Application

We developed a Streamlit-based interface to demonstrate the model and support interactive testing.

The application supports:

- Image upload.
- Video upload.
- Confidence threshold control.
- IoU/NMS threshold control.
- Bangladesh traffic detection module.
- Placeholder modules for breast cancer detection and drone detection.
- System configuration page.
- Display of supported classes and contributor details.

### 10.2 Workspace Preservation

The backup and staging workflow was designed to:

- Copy important scripts.
- Copy YAML files.
- Copy result CSV files.
- Generate performance plots.
- Package files into a compressed archive.
- Preserve important experiment artifacts for GitHub or future thesis work.

### 10.3 Cloud Backup and Resume Protocol

Cloud platforms such as Kaggle and Colab can disconnect during long runs. To reduce this risk, we used:

- Frequent checkpoint saving.
- Kaggle dataset backup callbacks.
- Google Drive storage for Colab outputs.
- Runtime re-injection of custom PyTorch modules.
- Resume training from `last.pt` when runs were interrupted.

This was especially important for DCFM_v2 because PyTorch must know the custom class definition before loading custom checkpoint weights.

### 10.4 Security Note

Any API token, Kaggle token, or cloud credential should not be stored in the thesis summary or repository. If a token was previously copied into notes, it should be revoked and regenerated from the provider dashboard.

---

## 11. Key Findings

### 11.1 Higher Resolution Improved Recall

Moving from 640px to 1024px improved recall on the harder Dhaka V6 dataset.

```text
Recall at 640px:   53.57%
Recall at 1024px:  67.90%
Improvement:       +14.33 percentage points
```

This happened because higher-resolution images preserve more detail for small objects such as cycles, pedestrians, and distant vehicles.

### 11.2 SAHI Patch Training Failed for This Dataset

SAHI-style patch training produced a major performance collapse.

```text
SAHI mAP@50:      0.74%
SAHI mAP@50-95:   0.16%
```

The likely reason is that slicing full traffic scenes into patches removes global context and cuts large objects into fragments. For our dataset, SAHI should be used carefully as an inference-time technique rather than the main training format.

### 11.3 Class Imbalance Remained a Major Challenge

The model performed strongly on frequent classes such as cars, rickshaws, CNGs, and buses, but rare classes such as cycles remained difficult.

This shows that architecture improvements alone cannot fully solve extreme data scarcity. More targeted data collection and augmentation are required.

### 11.4 CBAM Helped With Feature Focus

CBAM improved the model's ability to focus on relevant semantic and spatial regions. This was especially useful in cluttered scenes with overlapping vehicles and noisy backgrounds.

### 11.5 DCFM Improved Multi-Scale Sensitivity

DCFM and DCFM_v2 helped improve recall by giving the model better access to different receptive fields. This is useful because traffic objects appear at many different scales in the same frame.

---

## 12. Limitations

Our current work still has several limitations:

- The Dhaka V6 model has not yet reached the accuracy level of the earlier, easier conference-stage dataset.
- Precision dropped during high-resolution runs, even though recall improved.
- Minority classes such as cycles remain difficult because of limited examples.
- SAHI patch training failed when used directly as a training dataset.
- Local hardware limits restricted batch size and long high-resolution experimentation.
- Tracking and risk assessment are still future work, not final completed modules.

---

## 13. Current Status

The project has completed:

- Dataset conversion and validation.
- YOLO baseline training.
- CBAM attention integration.
- DCFM and DCFM_v2 experimentation.
- Standard YOLOv11m comparison.
- ICEFT 2026 conference acceptance.
- Dhaka V6 local dataset training.
- 1024px high-resolution experimentation.
- SAHI patch-training failure analysis.
- Streamlit application development.
- Backup and resume workflow design.

The strongest local Dhaka V6 result so far is the V6.5 high-resolution baseline:

```text
mAP@50:       49.78%
Recall:       67.90%
mAP@50-95:    33.12%
```

---

## 14. Future Work

### 14.1 Improve Minority-Class Detection

Next steps:

- Collect more examples of cycles, bikes, and pedestrians.
- Use targeted copy-paste augmentation for rare classes.
- Try class-balanced sampling.
- Test loss reweighting or focal-style loss options.
- Add more diverse lighting, weather, and road-condition examples.

### 14.2 Improve Architecture

Possible architecture directions:

- Add a P2 small-object detection head.
- Combine CBAM and DCFM carefully in a stable architecture.
- Test Transformer-style attention for global context.
- Use SAHI only during inference, not direct patch training.
- Compare YOLO26m against newer YOLO variants under the same dataset split.

### 14.3 Add Multi-Object Tracking

The next major system extension is tracking.

Recommended steps:

- Connect the best YOLO weights to ByteTrack.
- Assign object IDs across video frames.
- Track bounding-box centers over time.
- Measure motion paths in dense traffic.
- Analyze object interactions at intersections.

### 14.4 Add Traffic Risk Assessment

After tracking becomes stable, the project can move toward safety analysis.

Possible metrics:

- Time-to-Collision (TTC)
- Post-Encroachment Time (PET)
- Near-miss detection
- Conflict-zone analysis
- Vehicle interaction mapping

This would move the thesis from detection toward intelligent traffic safety analysis.

---

## 15. Final Summary

This thesis develops a YOLO-based traffic perception system for unstructured Bangladeshi roads. We started from dataset preparation and annotation conversion, then progressed through YOLO baselines, CBAM attention, DCFM multi-scale fusion, high-resolution local training, SAHI analysis, and deployment planning.

The project contributes:

- A structured Bangladeshi traffic detection workflow.
- Pascal VOC XML to YOLO annotation conversion.
- Dataset sanity checking and YAML preparation.
- Early 4-class and 9-class YOLO baselines.
- A peer-reviewed YOLOv11m-CBAM conference milestone.
- DCFM and DCFM_v2 multi-scale fusion experimentation.
- A harder Dhaka V6 local dataset benchmark.
- High-resolution 1024px training analysis.
- SAHI patch-training failure analysis.
- A Streamlit demonstration interface.
- A roadmap for ByteTrack tracking and traffic risk assessment.

Overall, our work provides a strong foundation for Bangladeshi traffic perception research. The current system is not only a detector, but also a research platform that can be extended toward tracking, risk estimation, and real-time ADAS applications for local road environments.
