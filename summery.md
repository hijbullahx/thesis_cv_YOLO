# Extended Thesis Summary: Bangladesh Traffic Perception System

**Institution:** International University of Business Agriculture and Technology (IUBAT)  
**Department:** Computer Science and Engineering  
**Research Area:** Computer Vision, Autonomous Systems, Advanced Driver Assistance Systems (ADAS), Intelligent Transportation Systems (ITS)  
**Project Eco-System Title:** Bangladesh Traffic Perception System, core component of the "Omars' Valley" autonomous infrastructure concept  
**Authors / Contributors:** Md. Taher Bin Omar Hijbullah and Md. Rony Mia  
**Core Architecture Base:** YOLOv11 / YOLO26m with custom PyTorch module extensions  
**Research Focus:** Object detection in dense, unstructured, non-lane-based Bangladeshi traffic scenes

---

## Executive Summary

In this thesis, we developed a YOLO-based traffic perception system for complex Bangladeshi road environments. Our main goal was to detect traffic objects in dense, unstructured, and non-lane-based scenes where standard traffic detection models often fail.

We worked through a full research pipeline: dataset preparation, annotation conversion, data validation, model training, architectural improvement, high-resolution experimentation, SAHI analysis, web deployment, and conference publication. Our work started from simple cloud-based baselines and gradually evolved into a larger local Dhaka traffic perception system trained on an RTX 4000 workstation.

Our first major academic milestone was the acceptance of our paper, **"Attention-Guided YOLOv11 for Object Detection in Unstructured Bangladeshi Traffic,"** at the **2026 International Conference on Engineering and Frontier Technologies (ICEFT 2026)** in Tangail, Bangladesh. This conference publication validated our early attention-guided YOLOv11-CBAM approach.

After that milestone, we continued improving the system because real Dhaka traffic scenes showed stronger occlusion, smaller road users, dense vehicle overlap, and more background confusion than ordinary benchmark datasets. We expanded the dataset to **28,159 valid images** with **289,506 annotated object instances**, reorganized the system into an 8-class Dhaka traffic framework, and tested 1024px high-resolution training across long local workstation runs of **21.58 hours** and **33.82 hours**.

This thesis represents our original effort as a team: we did not only train a model; we built a complete traffic perception research workflow for Bangladeshi road conditions.

---

## Contribution Snapshot

| Area | What We Contributed |
| --- | --- |
| Dataset Engineering | We prepared and validated a large Bangladeshi traffic detection dataset |
| Annotation Processing | We converted Pascal VOC XML annotations into normalized YOLO labels |
| Data Integrity | We used video-wise splitting to reduce train/test leakage |
| Model Architecture | We tested CBAM attention and DCFM multi-scale feature fusion modules |
| Experimentation | We ran cloud and local workstation experiments across multiple model versions |
| High-Resolution Training | We evaluated 1024px training to recover small-object features |
| Failure Analysis | We showed why SAHI patch training failed for this dataset |
| Deployment | We built a Streamlit interface for interactive model demonstration |
| Academic Output | We achieved peer-reviewed conference acceptance at ICEFT 2026 |

---

## 1. Problem Background

We focused on Bangladeshi traffic because local roads are visually and behaviorally different from structured lane-based traffic environments. In Dhaka and other dense urban areas, road scenes often contain:

- Rickshaws, CNG auto-rickshaws, cycles, bikes, buses, cars, trucks, and pedestrians in the same frame.
- Heavy occlusion between vehicles.
- Small and partially visible road users.
- Non-lane-based movement.
- Irregular road behavior.
- Dense background clutter from roadsides, shadows, signs, and urban objects.

Many existing traffic detection models are trained on structured datasets from more controlled road environments. When these models are applied to South Asian traffic, they often struggle because the object distribution, road-user behavior, and visual density are different.

Our thesis addresses this gap by building and evaluating a region-aware traffic object detection pipeline for Bangladeshi road conditions.

---

## 2. Our End-to-End Research Pipeline

```text
Raw Bangladeshi / Dhaka Traffic Frames
        |
        v
Pascal VOC XML Annotation Collection
        |
        v
XML to YOLO Conversion and Label Normalization
        |
        v
Dataset Cleaning, Sanity Checking, and Class Verification
        |
        v
Train / Validation / Test Split
        |
        +--> Early 80/20 split for cloud experiments
        +--> Strict 70/15/15 video-wise split for local V6 experiments
        |
        v
YOLO Baseline Training
        |
        +--> YOLO11n baseline
        +--> YOLOv11m baseline
        +--> YOLOv11m + CBAM
        +--> YOLOv11m + DCFM
        +--> YOLO26m local Dhaka framework
        |
        v
High-Resolution Training and Ablation Analysis
        |
        +--> 640px cloud runs
        +--> 1024px local workstation runs
        +--> SAHI patch-training experiment
        |
        v
Evaluation, Web Demonstration, Tracking Roadmap, and Thesis Documentation
```

This pipeline shows the actual sequence of our work. We moved from raw data to model design, then to experiments, failure analysis, deployment, and academic presentation.

---

## 3. Chronological Log of Our System Iterations

### 3.1 Iteration 1: Baseline Light Model Formulation (V1)

Our first goal was to build a simple proof-of-concept detector using a limited number of traffic classes.

| Item | Details |
| --- | --- |
| Objective | Establish our first custom-scaled single-stage detector |
| Compute Environment | Kaggle Kernel, single NVIDIA Tesla T4 |
| Dataset | 4 balanced classes: car, bus, truck, rickshaw |
| Valid Image-Label Pairs | 23,564 |
| Split | 16,494 train / 4,712 validation / 2,358 test |
| Training Setup | 640x640 image size, batch 16, 50 epochs |
| Runtime | Approximately 2.77 hours |

**Results**

```text
Precision:     72.46%
Recall:        70.55%
mAP@50:        76.88%
mAP@50-95:     54.62%
```

**What we learned:** The first baseline gave us strong initial numbers, but it also exposed background false positives and truck detection failures. We found that long-tail classes were harder to detect, especially when their appearance overlapped with background road patterns.

---

### 3.2 Iteration 2: Class Expansion and Data Formatting (V2)

After the first baseline, we expanded the system from 4 classes to 9 local road-user categories.

| Item | Details |
| --- | --- |
| Objective | Expand detection from 4 classes to 9 traffic categories |
| Main Script | `fix_labels.py` |
| Conversion Task | Pascal VOC XML to YOLO normalized labels |
| Processed Files | 23,648 |
| Parsing Success | Approximately 99.8% |
| Split Strategy | 80% train / 20% validation |
| Archive | `bangladesh_yolo_ready.zip`, approximately 1,605.53 MB |

This stage was one of our most important dataset engineering steps. We created a reusable annotation conversion pipeline and prepared the dataset for larger multi-class training.

---

### 3.3 Iteration 3: Core CBAM Layer Integration (V3)

We then introduced CBAM attention to help the model focus on important traffic features and suppress cluttered background information.

| Item | Details |
| --- | --- |
| Objective | Add attention to improve feature selection in dense traffic scenes |
| Architecture | YOLOv11m + CBAM |
| CBAM Placement | Deep P5 feature layer |
| Compute Environment | Kaggle DDP, 2x NVIDIA Tesla T4 |
| Configuration | 9 classes, 640x640 image size, batch 32, 50 epochs |
| Scheduler | Cosine learning rate scheduler |
| Runtime | 5.584 hours |

**Results**

```text
Precision:     67.10%
Recall:        72.00%
mAP@50:        74.90%
mAP@50-95:     52.00%
```

**Class-Level Observation**

```text
Car mAP:       93.30%
Rickshaw mAP: 90.40%
Cycle mAP:    31.50%
```

**What we learned:** CBAM helped the model focus better on important visual regions, especially for common and distinctive objects. However, cycle performance remained low, which showed us that attention alone cannot solve severe class imbalance.

---

### 3.4 Iteration 4: Targeted CBAM Fine-Tuning (V4)

We performed a separate fine-tuning stage to improve weaker classes.

| Item | Details |
| --- | --- |
| Objective | Fine-tune attention features for underrepresented classes |
| Compute Environment | Kaggle multi-GPU, 2x Tesla T4 |
| Training | 30 specialized epochs |
| Log File | `resultss.csv` |

**Results**

```text
Global mAP@50:      75.13%
Mini-Truck mAP:     70.00%
Truck mAP:          50.00%
Cycle mAP:          32.00%
```

**What we learned:** Fine-tuning improved mini-truck and truck performance, but cycle detection stayed weak. This confirmed that rare classes need more samples, targeted augmentation, or class balancing.

---

### 3.5 Iteration 5: Reference Medium Baseline (V5)

We trained an unmodified YOLOv11m model to create a fair reference baseline.

| Item | Details |
| --- | --- |
| Objective | Establish a standard YOLOv11m comparison baseline |
| Architecture | Standard YOLOv11m |
| Compute Environment | Kaggle DDP, 2x Tesla T4 |
| Training Images | 18,919 |
| Validation Frames | 4,729 |
| Annotated Validation Targets | 52,941 |
| Training Length | 50 epochs |
| Runtime | 6.928 hours |

**Results**

```text
Precision:     69.70%
Recall:        71.40%
mAP@50:        75.10%
mAP@50-95:     52.80%
```

This gave us a stable baseline for comparing our attention-guided and multi-scale model variants.

---

### 3.6 Iteration 6: Dynamic Multi-Scale Kernel Fusion (V5.2)

We designed and integrated the Dynamic Channel Fusion Module (DCFM) to handle scale variation in traffic scenes.

| Item | Details |
| --- | --- |
| Objective | Improve detection across different object sizes |
| Module | Dynamic Channel Fusion Module (DCFM) |
| Configuration | `yolov11m-dcfm.yaml` |
| Scale | depth = 0.50, width = 1.00, max channels = 512 |

**DCFM Concept**

```text
Input Feature Map
        |
        +--> 3x3 convolution branch
        +--> 5x5 convolution branch
        +--> Dilated convolution branch
        |
        v
Softmax attention weighting
        |
        v
Dynamic multi-scale fused output
```

**Results**

```text
Global recall:  73.08%
mAP@50:         74.76%
```

**What we learned:** DCFM pushed the model toward stronger recall, which is valuable for traffic safety because missed detections can be more dangerous than extra detections in ADAS-style systems.

---

## 4. Our First Public Milestone: ICEFT 2026

We used our 640px YOLOv11m-CBAM findings to prepare and submit our first conference paper:

**"Attention-Guided YOLOv11 for Object Detection in Unstructured Bangladeshi Traffic"**

| Item | Details |
| --- | --- |
| Conference | 2026 International Conference on Engineering and Frontier Technologies (ICEFT 2026) |
| Location | Tangail, Bangladesh |
| Presentation Date | January 28, 2026 |
| Conference Site | <https://icefront.mbstu.ac.bd/> |
| Document Record | `Attention_Guided_YOLOv11_for_Object_Detection_in_Unstructured_Bangladeshi_Traffic_c_1.pdf` |

**Published Peer-Reviewed Baselines**

```text
Standard YOLOv11m baseline:     75.1% mAP
Custom YOLOv11m-CBAM model:     75.7% mAP
Reported edge latency:          4.7 ms / frame
```

This was an important point in our research journey because our work was reviewed and accepted outside our university project environment.

---

## 5. Our Pivot to Local Optimization: Dhaka V6 Dataset

After the conference-stage work, we tested our models more deeply on realistic Dhaka traffic scenes. These scenes exposed serious difficulties: heavy occlusion, close vehicle spacing, smaller non-motorized road users, and strong background confusion.

To address this, we reorganized the project into an 8-class local framework called **Dhaka_Traffic_Local Version 6**.

### 5.1 Dataset Statistics

```text
Total valid images:                 28,159
Total annotated object instances:   289,506
Average objects per frame:          Approximately 10.5
Dataset split:                      70% train, 15% validation, 15% test
Target classes:                     8
```

### 5.2 Why We Used Video-Wise Splitting

We used a strict video-wise cohort split so that frames from the same video stream were not randomly mixed across train, validation, and test sets.

This decision made our evaluation more trustworthy because nearby frames often contain:

- The same road background.
- The same camera angle.
- Similar lighting.
- The same vehicles across consecutive frames.
- Similar object positions.

Without this split, the model could appear stronger than it really is by memorizing nearly identical frames.

---

## 6. Our V6 Local Experiment Series

### 6.1 Iteration 7: Off-the-Shelf Dhaka Baseline (V6.1)

We first trained a baseline model on the local Dhaka dataset at standard 640px resolution.

| Item | Details |
| --- | --- |
| Objective | Establish a standard baseline on our Dhaka dataset |
| Compute Environment | Google Colab, Tesla T4 |
| Architecture | Intermediate YOLO26m base |
| Image Size | 640px |
| Training Length | 50 epochs |
| Runtime | Approximately 4.20 hours |

**Result**

```text
Final mAP@50: 48.20%
```

**Severe Local Vehicle Failure Profiles**

```text
Rickshaw validation mAP@50:        15.1%
Pedestrian / people mAP@50:         8.1%
Bicycle / cycle mAP@50:             6.1%
```

**What we learned:** The local Dhaka dataset was much harder than our earlier dataset. Small and non-motorized classes suffered heavily under dense street crowding.

---

### 6.2 Iteration 8: Local 1024px Resolution Marathon (V6.2 - V6.4)

We increased the image size to 1024px to preserve more detail for small road users.

| Item | Details |
| --- | --- |
| Objective | Recover small-object features using higher input resolution |
| Image Size | 1024px |
| Augmentation | Mosaic 1.0 and Copy-Paste 0.3 |
| Compute Environment | Local workstation, NVIDIA RTX 4000, 8GB VRAM |
| Runtime | 21.58 hours, 77,707 seconds continuous run |
| Peak Epoch | Epoch 7 |

**Results**

```text
mAP@50:        49.67%
Global recall: 64.01%
```

**What we learned:** High-resolution training preserved more object detail and helped reduce the small-object vanishing problem.

---

### 6.3 Iteration 9: High-Resolution Baseline Peak (V6.5)

We then pushed our local hardware further to define the strongest high-resolution baseline.

| Item | Details |
| --- | --- |
| Objective | Secure our strongest high-resolution baseline |
| Architecture | YOLO26m base |
| Image Size | 1024px |
| Compute Environment | Local NVIDIA RTX 4000 workstation |
| Training Length | 50 complete epochs |
| Runtime | 33.82 hours, 121,738 seconds continuous run |
| Peak mAP Epoch | Epoch 14 |

**Peak Results**

```text
Peak global mAP@50:              49.78%
Global box recall:               67.90%
Strict localization mAP@50-95:   33.12%
```

**What we learned:** This was our strongest V6 high-resolution baseline. The 1024px setting improved recall and helped preserve small road-user features, but the model plateaued just below 50% mAP@50 because of background confusion, dense crowding, and class imbalance.

---

### 6.4 Iteration 10: SAHI Patch-Training Failure (V6_SAHI)

We tested whether patch-wise slicing could replace full-frame high-resolution training.

| Item | Details |
| --- | --- |
| Objective | Test SAHI-style patch training as an alternative to full-frame 1024px training |
| Script | `sahi_slicer.py` |
| Patch Size | 640x640 overlapping windows |
| Runtime | 15.48 hours, 55,718 seconds |

**Results**

```text
Global mAP@50:                   0.74%
Global mAP@50-95:                0.16%
Validation classification loss:  7.18
```

**What we learned:** This failure became an important research finding. Patch slicing damaged the global scene context needed for traffic detection. Large vehicles were split across patch boundaries, and the model learned partial object fragments instead of complete traffic objects.

Our conclusion is that SAHI should be used mainly as an inference-time or post-processing technique on full-frame trained weights, not as the main training format for our dataset.

---

## 7. Main Experimental Comparison

| Metric | V1 Baseline | V3 CBAM | V5 YOLOv11m | V6.1 Dhaka Base | V6.4 Local 1024px | V6.5 High-Res Peak | V6 SAHI Patch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model | YOLO11n | YOLOv11m-CBAM | YOLOv11m | YOLO26m | YOLO26m | YOLO26m | YOLO26m + SAHI |
| Classes | 4 | 9 | 9 | 8 | 8 | 8 | 8 |
| Image Size | 640 | 640 | 640 | 640 | 1024 | 1024 | 1024 sliced |
| Platform | Kaggle | Kaggle | Kaggle | Colab | Local | Local | Local |
| GPU | Tesla T4 | 2x Tesla T4 | 2x Tesla T4 | Tesla T4 | RTX 4000 | RTX 4000 | RTX 4000 |
| Runtime | 2.77 h | 5.58 h | 6.93 h | 4.20 h | 21.58 h | 33.82 h | 15.48 h |
| Precision | 72.46% | 67.10% | 69.70% | 44.95% | 46.22% | 36.39% | 2.55% |
| Recall | 70.55% | 72.00% | 71.40% | 53.57% | 50.81% | 67.90% | 4.26% |
| mAP@50 | 76.88% | 74.90% | 75.10% | 48.20% | 49.67% | 49.78% | 0.74% |
| mAP@50-95 | 54.62% | 52.00% | 52.80% | 31.55% | 30.79% | 33.12% | 0.16% |

---

## 8. Architecture Improvements We Tested

### 8.1 CBAM: Convolutional Block Attention Module

We used CBAM to help the model focus on important channels and spatial regions.

| CBAM Stage | Purpose |
| --- | --- |
| Channel Attention | Learns which feature channels are most important |
| Spatial Attention | Learns where important objects or object parts are located |

```python
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)
```

CBAM was useful for our traffic scenes because vehicles are frequently close together, visually similar, and partially hidden.

### 8.2 DCFM: Dynamic Channel Fusion Module

We designed DCFM to handle scale variation across small cycles, pedestrians, rickshaws, buses, and trucks.

| Branch | Role |
| --- | --- |
| `3x3` convolution | Captures compact local features |
| `5x5` convolution | Captures broader local context |
| Dilated convolution | Increases receptive field without excessive downsampling |

```python
class DCFM(nn.Module):
    def __init__(self, c1, c2):
        super(DCFM, self).__init__()
        self.conv_small = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False)
        self.conv_large = nn.Conv2d(c1, c2, kernel_size=5, padding=2, bias=False)
        self.conv_dilated = nn.Conv2d(c1, c2, kernel_size=3, padding=2, dilation=2, bias=False)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        weights = self.attention(x)
        out_small = self.conv_small(x)
        out_large = self.conv_large(x)
        out_dilated = self.conv_dilated(x)
        out = (
            out_small * weights[:, 0:1, :, :]
            + out_large * weights[:, 1:2, :, :]
            + out_dilated * weights[:, 2:3, :, :]
        )
        return self.act(self.bn(out))
```

### 8.3 YOLO26m Combined Architecture

We used the combined YOLO26m architecture as the base for our local Dhaka traffic experiments.

```yaml
nc: 8
scales:
  m: [0.67, 0.75, 768]
```

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `depth` | `0.67` | Controls model depth |
| `width` | `0.75` | Controls channel width |
| `max_channels` | `768` | Limits maximum channel size |

---

## 9. Our Project Repository Layout

```text
D:\Hijbullah\Thesis\
|-- Master_Dataset_Local\               # Source root of localized V6 data
|   |-- images\
|   |   |-- train\                      # 18,919 valid training frames
|   |   |-- val\                        # 4,729 validation frames
|   |   `-- test\                       # 4,511 testing scenes
|   `-- labels\
|       |-- train\                      # Aligned 8-class YOLO labels
|       |-- val\                        # Normalized bounding-box labels
|       `-- test\                       # Ground-truth evaluation labels
|
|-- Ablation_Runs\
|   |-- Dhaka_CBAM_Only\
|   |-- Dhaka_DCFM_Only\
|   `-- Dhaka_Combined\
|
|-- High_Res_Runs\
|   `-- Dhaka_Combined_1024px\
|
|-- SAHI_Dataset\
|   |-- images\train\
|   `-- labels\train\
|
|-- Qualitative_Analysis\
|-- Git_Staging_Master\
|-- download_data.py
|-- fix_labels.py
|-- fix_yamls.py
|-- inject.py
|-- patch_tasks.py
|-- dataset_sanity_check.py
|-- model_vision_check.py
|-- sahi_slicer.py
|-- marathon.py
|-- high_res_marathon.py
|-- sahi_marathon.py
|-- risk_tracker.py
|-- setup_files.py
|-- app.py
|-- Dhaka_Traffic_Local.yaml
`-- Dhaka_Traffic_SAHI.yaml
```

We organized the project this way to keep datasets, training runs, high-resolution logs, SAHI experiments, qualitative outputs, source code, and deployment files separated.

---

## 10. Production Software We Built

### 10.1 Automated Workspace Shield and Preservation Script

Main file:

```text
setup_files.py
```

We created this script to preserve important project artifacts and prepare a backup archive.

The script handles:

- Project path validation.
- Copying critical code files.
- Copying YAML configuration files.
- Copying training result CSV files.
- Generating mAP convergence plots.
- Creating a compressed `.zip` backup archive.

### 10.2 Streamlit Web Application Interface

Main file:

```text
app.py
```

We built a Streamlit dashboard to demonstrate our model and make the system easier to test interactively.

The application supports:

- Image upload.
- Video upload.
- Confidence threshold adjustment.
- IoU/NMS threshold adjustment.
- Bangladesh traffic detection demonstration.
- Breast cancer detection placeholder panel.
- Drone detection placeholder panel.
- System configuration panel.
- Supported class and contributor information display.

This application helped us turn the thesis from an offline training project into a practical demonstration system.

---

## 11. Key Findings From Our Experiments

### 11.1 High Resolution Improved Small-Object Recall

We found that increasing the input resolution from 640px to 1024px improved recall in the local Dhaka dataset.

```text
V6.1 recall at 640px:       53.57%
V6.5 recall at 1024px:      67.90%
Improvement:                +14.33 percentage points
```

This happened because 1024px training preserved more visual information for small and distant road users such as cycles, bikes, pedestrians, and rickshaws.

### 11.2 SAHI Training Was Not Suitable for Our Dataset

Our SAHI patch-training experiment caused a severe metric collapse.

```text
SAHI mAP@50:      0.74%
SAHI mAP@50-95:   0.16%
```

This failure was still valuable. It showed us that slicing full traffic scenes into patches can remove the context needed for correct object learning.

### 11.3 Class Imbalance Remained a Major Limitation

Our dataset contains strong class imbalance between common and rare classes.

```text
Cars:     Approximately 14,000 instances
Cycles:   Approximately 150 instances
```

We found that attention modules can improve feature focus, but they cannot fully compensate for extreme data scarcity. Future work should include more minority-class samples, targeted augmentation, and class-balanced training strategies.

---

## 12. Current Research Status

At this stage, we have completed the data engineering and baseline validation phase. We have also documented the transition from early cloud prototypes to high-resolution local workstation training.

Completed work:

- We built a four-class baseline model.
- We expanded the dataset into a nine-class cloud experiment setup.
- We converted Pascal VOC annotations into YOLO labels.
- We integrated CBAM attention.
- We designed and tested DCFM multi-scale fusion.
- We compared against a standard YOLOv11m baseline.
- We achieved ICEFT 2026 paper acceptance.
- We built the Dhaka V6 8-class dataset framework.
- We used video-wise dataset splitting.
- We ran 1024px local high-resolution training.
- We analyzed SAHI patch-training failure.
- We built a Streamlit web application.
- We created backup and packaging automation.

Current technical status:

| Strength | Remaining Challenge |
| --- | --- |
| Our dataset pipeline is validated | Minority classes still need more examples |
| Our 1024px training improved recall | Precision still needs improvement |
| Our SAHI limitation is clearly documented | Inference-time SAHI still needs careful testing |
| Our web demo exists | Deployment can be polished further |
| Our conference milestone is achieved | Thesis and journal-level extension can be developed |

---

## 13. Upcoming Milestones

### Phase 3: NMS-Free Tracking Integration

Our next major step is tracking. We plan to:

- Route the best YOLO26m combined weights into a ByteTrack inference pipeline.
- Extract continuous bounding-box centroids across video frames.
- Assign persistent object IDs.
- Analyze movement paths in dense Dhaka traffic.

### Phase 4: Vector-Based Street Risk Assessment

After tracking becomes stable, we plan to extend the system toward traffic risk analysis.

Possible risk metrics include:

- Time-to-Collision (TTC).
- Post-Encroachment Time (PET).
- Near-miss detection.
- Vehicle interaction mapping.
- Risk benchmarking against high-danger Dhaka traffic sequences.

This will move our work beyond object detection and toward intelligent transportation safety analysis.

---

## 14. Our Original Effort and Research Value

This thesis required a large amount of direct work from us. The most important original efforts include:

- We prepared a large local traffic dataset.
- We processed tens of thousands of annotation files.
- We built XML-to-YOLO conversion scripts.
- We validated annotation quality before training.
- We designed a video-wise split to reduce leakage.
- We ran experiments on Kaggle, Google Colab, and a local RTX 4000 workstation.
- We implemented and tested CBAM attention.
- We designed and tested DCFM multi-scale fusion.
- We conducted long 1024px workstation training runs lasting more than 21 and 33 hours.
- We investigated a failed SAHI training direction and turned it into a useful thesis finding.
- We built a Streamlit demonstration interface.
- We prepared our work for peer-reviewed conference publication.

This record shows our complete thesis journey: problem discovery, dataset engineering, model design, training, failure analysis, deployment, and academic communication.

---

## Final Thesis Summary

In this thesis, we developed a YOLO-based object detection system for unstructured Bangladeshi traffic. Our work addresses dense object layouts, occlusion, small-object detection, class imbalance, and regional traffic diversity.

We contributed a validated Bangladeshi traffic detection workflow, custom attention and multi-scale model components, multiple experimental comparisons, high-resolution local training analysis, SAHI failure analysis, a Streamlit demonstration application, and a peer-reviewed conference milestone.

Overall, our system provides a strong foundation for Bangladeshi traffic perception research. In future work, we can extend it toward object tracking, traffic risk assessment, and real-time ADAS applications for complex local road environments.
