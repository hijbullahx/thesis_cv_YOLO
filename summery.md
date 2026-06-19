# Thesis Summary: Bangladesh Traffic Perception System

**Institution:** International University of Business Agriculture and Technology (IUBAT)  
**Department:** Computer Science and Engineering  
**Research Area:** Computer Vision, Autonomous Systems, Advanced Driver Assistance Systems (ADAS), and Intelligent Transportation Systems (ITS)  
**Project Title:** Bangladesh Traffic Perception System  
**Core Model Family:** YOLOv11 / YOLO26m with attention-based architectural improvements  

---

## 1. Project Overview

This thesis focuses on building an object detection system for unstructured Bangladeshi traffic scenes. Traffic in Bangladesh, especially in dense urban areas such as Dhaka, contains high vehicle diversity, heavy occlusion, irregular road behavior, non-lane-based movement, and many small or partially visible objects.

Standard traffic detection models are usually trained on structured road environments. As a result, they often struggle when applied to South Asian traffic conditions, where vehicles such as rickshaws, CNG auto-rickshaws, cycles, buses, trucks, and pedestrians appear in dense and unpredictable arrangements.

The proposed system improves YOLO-based traffic detection by integrating attention and multi-scale feature fusion modules. The goal is to improve detection performance for complex, crowded, and region-specific traffic objects.

---

## 2. System Architecture

The overall pipeline converts raw traffic video data into a structured YOLO training dataset, validates all annotations, trains multiple model variants, and prepares the best models for web-based demonstration and future deployment.

```text
Raw Dhaka Traffic Video Frames
        |
        v
Pascal VOC XML Annotation Processing
        |
        v
YOLO Label Conversion and Normalization
        |
        v
Strict Video-Wise 70/15/15 Dataset Split
        |
        v
Dataset Sanity Checking and Annotation Validation
        |
        v
Model Training and Ablation Experiments
        |
        +--> YOLOv11m + CBAM
        +--> YOLOv11m + DCFM
        +--> YOLO26m Combined Architecture
        |
        v
Evaluation, Error Analysis, and Web Application Deployment
```

The system is designed to support both local workstation training and cloud-based experimentation.

---

## 3. Dataset Preparation

The dataset was prepared using regional traffic footage and benchmark traffic data. The main objective was to create a dataset that better represents Bangladeshi road conditions.

### 3.1 Dataset Sources

The dataset combines:

- Bangladeshi traffic footage from the Mendeley Bangladeshi Traffic Flow Dataset.
- Additional traffic object data inspired by public benchmark datasets such as VisDrone, mainly to improve small-object detection behavior.

### 3.2 Dataset Statistics

```text
Total valid images:                 28,159
Total annotated object instances:   289,506
Average objects per frame:          Approximately 10.5
Dataset split:                      70% train, 15% validation, 15% test
```

The dataset contains eight target classes representing common road users in Bangladeshi traffic scenes. These include rickshaws, CNG auto-rickshaws, cars, buses, trucks, cycles, bikes, and pedestrians or people, depending on the final class naming used in the training YAML file.

### 3.3 Video-Wise Dataset Split

A strict video-wise split was used instead of random frame shuffling.

This is important because consecutive frames from traffic videos often contain nearly identical backgrounds, vehicles, lighting, and camera angles. If frames from the same video are randomly placed into both training and testing sets, the model can memorize visual patterns instead of learning general object features.

The video-wise split helps reduce data leakage and gives a more realistic evaluation of model performance.

---

## 4. Annotation Conversion and Validation

The original annotations were stored in Pascal VOC XML format. These bounding boxes were converted into YOLO format using normalized coordinates.

### 4.1 Pascal VOC to YOLO Conversion

YOLO requires each annotation to follow this format:

```text
class_id x_center y_center width height
```

All coordinates must be normalized between `0.0` and `1.0`.

```python
def normalize_pascal_xml_to_yolo(x_min, x_max, y_min, y_max, img_width, img_height):
    x_center = (x_min + x_max) / (2.0 * img_width)
    y_center = (y_min + y_max) / (2.0 * img_height)
    bbox_width = (x_max - x_min) / img_width
    bbox_height = (y_max - y_min) / img_height
    return float(x_center), float(y_center), float(bbox_width), float(bbox_height)
```

### 4.2 Dataset Sanity Checking

A validation script was used to check annotation quality before training. The script verifies:

- Class IDs are inside the valid class range.
- Bounding-box coordinates are normalized correctly.
- Width and height values are valid.
- Annotation files are not corrupted.

Example validation logic:

```python
if class_id >= len(class_names) or class_id < 0:
    raise ValueError(f"Invalid class ID detected: {class_id}")

if x_center > 1.0 or y_center > 1.0 or box_w > 1.0 or box_h > 1.0:
    raise ValueError("Bounding-box coordinates are outside the normalized range [0, 1].")
```

This validation step helped prevent training failures caused by incorrect labels.

---

## 5. Local Project Structure

The local project was organized into separate folders for datasets, training runs, high-resolution experiments, SAHI experiments, qualitative outputs, and deployment files.

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

This structure keeps the main dataset, experimental outputs, and deployment files separated and easier to manage.

---

## 6. Proposed Model Improvements

Two major custom modules were introduced to improve detection in dense and unstructured traffic scenes:

1. **CBAM: Convolutional Block Attention Module**
2. **DCFM: Dynamic Channel Fusion Module**

These modules were integrated into YOLO-based model architectures to improve attention, feature selection, and multi-scale representation.

---

## 7. CBAM: Convolutional Block Attention Module

CBAM improves feature representation by helping the model focus on the most important channels and spatial regions.

It contains two attention stages:

- **Channel Attention:** Learns which feature channels are most important.
- **Spatial Attention:** Learns where important objects or object parts are located in the feature map.

This is useful in Bangladeshi traffic scenes because vehicles are often close together, partially hidden, or visually similar.

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

CBAM was inserted into deeper feature layers where high-level semantic information is available. This allows the network to better separate meaningful vehicle features from noisy background regions.

---

## 8. DCFM: Dynamic Channel Fusion Module

The Dynamic Channel Fusion Module was designed to handle scale variation. In real traffic scenes, objects appear at different sizes because of distance, camera angle, and occlusion.

DCFM uses three parallel convolution paths:

- A `3x3` convolution for small receptive fields.
- A `5x5` convolution for larger local context.
- A dilated convolution for wider spatial coverage.

The module then uses softmax-based attention weights to combine these branches dynamically.

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

This module helps the model adapt to both small objects, such as cycles or distant pedestrians, and larger objects, such as buses and trucks.

---

## 9. Combined YOLO26m Architecture

The combined architecture integrates both CBAM and DCFM into the YOLO26m model structure.

The design goal was to improve:

- Small-object detection.
- Dense traffic object separation.
- Multi-scale feature extraction.
- Detection under occlusion.
- Robustness in unstructured road scenes.

The model configuration uses an eight-class detection head:

```yaml
nc: 8
scales:
  m: [0.67, 0.75, 768]
```

In this configuration:

- `depth = 0.67`
- `width = 0.75`
- `max_channels = 768`

These scaling values control the model size and channel width while keeping the architecture computationally manageable.

---

## 10. Experimental Progress

Several model versions were trained and compared across different environments, including Kaggle, Google Colab, and a local RTX 4000 workstation.

### 10.1 Main Experimental Results

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

## 11. Key Findings

### 11.1 Effect of Higher Resolution

Increasing the input resolution from `640px` to `1024px` improved recall significantly.

```text
Recall at 640px:    53.57%
Recall at 1024px:   67.90%
Improvement:        +14.33 percentage points
```

This improvement happened because small objects retain more visual detail at higher resolution. Objects such as cycles, bikes, pedestrians, and distant vehicles are easier for the model to detect when downsampling destroys less information.

The best strict localization score was also achieved at high resolution:

```text
Best mAP@50-95: 33.12%
```

### 11.2 SAHI Patch-Training Problem

SAHI-style slicing was tested as a training strategy, but it caused a major performance collapse:

```text
SAHI mAP@50:      0.74%
SAHI mAP@50-95:   0.16%
```

The likely reason is that slicing full traffic frames into smaller patches removes important global context. Large objects such as buses and trucks may be cut into partial pieces, while dense background regions may appear without enough surrounding information.

The conclusion is that SAHI should be used mainly as a post-processing or inference-time technique, not as the main training format for this dataset.

### 11.3 Class Imbalance

The dataset contains strong class imbalance. Some classes have many examples, while others are rare.

Example:

```text
Cars:     Approximately 14,000 instances
Cycles:   Approximately 150 instances
```

This imbalance causes the model to learn frequent classes more easily than rare classes. As a result, minority classes such as cycles remain difficult to detect.

Attention modules can improve feature focus, but they cannot fully solve extreme data scarcity. Additional dataset balancing, targeted augmentation, and more minority-class samples are required.

---

## 12. Conference Acceptance

The research achieved a peer-reviewed conference acceptance milestone.

**Conference:** 2026 International Conference on Engineering and Frontier Technologies (ICEFT 2026)  
**Location:** Tangail, Bangladesh  
**Presentation Date:** January 28, 2026  
**Conference Website:** <https://icefront.mbstu.ac.bd/>  
**Document Record:** `Attention_Guided_YOLOv11_for_Object_Detection_in_Unstructured_Bangladeshi_Traffic_c_1.pdf`

### 12.1 Published Research Summary

The accepted work proposes an attention-guided YOLOv11 architecture for object detection in unstructured Bangladeshi traffic. The model uses CBAM attention to improve feature selection in dense and irregular traffic scenes.

The paper reports that the attention-optimized model improves detection performance compared with the baseline YOLOv11m model while maintaining real-time inference capability.

Reported publication values:

```text
Baseline YOLOv11m mAP:        75.1%
Attention-guided model mAP:   75.7%
Reported latency:             4.7 ms per frame
```

---

## 13. Web Application Deployment

A Streamlit-based web interface was developed to demonstrate model inference and allow interactive testing.

The web application supports:

- Image and video upload.
- Confidence threshold adjustment.
- IoU/NMS threshold adjustment.
- Multiple research domains.
- Traffic object detection demonstration.
- Presentation of supported object classes and contributor details.

Main application file:

```text
app.py
```

The application includes modules for:

- Bangladesh traffic detection.
- Breast cancer detection interface placeholder.
- Drone detection interface placeholder.
- System configuration panel.

---

## 14. Automation and Backup Script

A setup and packaging script was created to collect important project files, generate result charts, and prepare a compressed archive for backup or GitHub staging.

Main script:

```text
setup_files.py
```

The script handles:

- Project path validation.
- Copying critical source files and YAML configuration files.
- Copying training result CSV files.
- Generating mAP trend plots.
- Creating a compressed `.zip` backup archive.

This improves reproducibility and helps preserve important experiment files.

---

## 15. Current Research Status

The project has completed several important stages:

- Dataset conversion and validation.
- YOLO-based baseline training.
- CBAM attention integration.
- DCFM multi-scale feature fusion design.
- Combined architecture experimentation.
- High-resolution training analysis.
- SAHI patch-training analysis.
- Web interface development.
- Conference paper acceptance.

The current best high-resolution experiment achieved stronger recall, but precision and minority-class performance still need improvement.

---

## 16. Future Research Roadmap

### Phase 1: Improve Minority-Class Performance

Recommended next steps:

- Add more cycle, bike, and pedestrian samples.
- Use targeted data augmentation for rare classes.
- Apply copy-paste augmentation carefully.
- Test class-balanced sampling or loss reweighting.

### Phase 2: Improve Tracking

The next major system extension is object tracking.

Recommended approach:

- Connect the best YOLO weights with ByteTrack.
- Extract object IDs across frames.
- Track bounding-box centers over time.
- Analyze object movement paths in dense traffic.

### Phase 3: Risk Assessment

After tracking is stable, the system can be extended toward traffic risk analysis.

Possible risk metrics:

- Time-to-Collision (TTC)
- Post-Encroachment Time (PET)
- Near-miss detection
- Vehicle interaction mapping

This would move the project beyond object detection and toward intelligent traffic safety analysis.

---

## 17. Final Summary

This thesis develops a YOLO-based object detection system for unstructured Bangladeshi traffic. The work addresses major challenges such as dense object layouts, occlusion, small-object detection, class imbalance, and regional traffic diversity.

The project contributes:

- A curated Bangladeshi traffic detection dataset.
- A validated Pascal VOC to YOLO conversion pipeline.
- A strict video-wise dataset split to reduce leakage.
- Custom CBAM and DCFM modules for attention and multi-scale fusion.
- Multiple YOLO-based experimental comparisons.
- High-resolution training analysis.
- SAHI failure analysis for patch-based training.
- A Streamlit demonstration application.
- A peer-reviewed conference publication milestone.

Overall, the system provides a strong foundation for Bangladeshi traffic perception research and can be extended toward tracking, risk assessment, and real-time ADAS applications.
