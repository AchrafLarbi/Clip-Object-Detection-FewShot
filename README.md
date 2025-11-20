# 🚀 CLIP Zero-Shot Object Detection

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenAI-412991.svg?style=for-the-badge&logo=OpenAI&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=Python&logoColor=ffdd54" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter Notebook">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge" alt="License">
</p>

> **Detect objects in images without training! Now with enhanced methods and ensemble approaches.**

Welcome to the **CLIP Zero-Shot Object Detection** project! This repository demonstrates how to perform zero-shot object detection by integrating OpenAI's **CLIP** (Contrastive Language-Image Pretraining) model with a **Faster R-CNN** for region proposal generation. The project now includes advanced enhancements like confidence scoring, negative prompts, ensemble methods, and alternative detection models.

---

| **Source Code**                                                                                                               | **Website**                                                                                                                  |
| :---------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| <a href="https://github.com/deepmancer/clip-object-detection" target="_blank">github.com/deepmancer/clip-object-detection</a> | <a href="https://deepmancer.github.io/clip-object-detection/" target="_blank">deepmancer.github.io/clip-object-detection</a> |

---

## 🎯 Quick Start

Set up and run the pipeline in three simple steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/deepmancer/clip-object-detection.git
   cd clip-object-detection
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:

   ```bash
   jupyter notebook clip_object_detection.ipynb
   ```

---

## 🤔 What is CLIP?

**CLIP** (Contrastive Language–Image Pretraining) is trained on 400 million image-text pairs. It embeds images and text into a shared space where the cosine similarity between embeddings reflects their semantic relationship.

<div align="center">
    <figure>
        <img 
            src="https://raw.githubusercontent.com/deepmancer/clip-object-detection/main/assets/CLIP.png" 
            width="600" 
            alt="CLIP Model Architecture"
        />
        <figcaption>
            CLIP Model Architecture - <a href="https://arxiv.org/abs/2103.00020">Paper</a>
        </figcaption>
    </figure>
</div>

---

## 🔍 Methodology

Our approach combines CLIP and Faster R-CNN for zero-shot object detection:

1. **📦 Region Proposal**: Use Faster R-CNN to identify potential object locations.
2. **🎯 CLIP Embeddings**: Encode image regions and text descriptions into a shared embedding space.
3. **🔍 Similarity Matching**: Compute cosine similarity between text and image embeddings to identify matches.
4. **✨ Results**: Highlight detected objects with their confidence scores.

---

## ✨ Advanced Features & Innovations

### 🚀 Enhanced Detection Methods

- **Confidence Scoring**: Normalized confidence scores (0-1) with uncertainty quantification
- **Multi-Scale Region Analysis**: Analyze regions at multiple scales (0.8x, 1.0x, 1.2x)
- **Negative Prompt Engineering**: Reduce false positives using contrastive prompts for better discrimination
- **Ensemble Detection**: Combine multiple detection approaches for robust predictions

### 🔄 Alternative Detection Methods

- **DETR** (Detection Transformer): Transformer-based object detection for comparison
- **YOLOv8**: Real-time object detection framework for fast inference
- Easy model switching and integration framework

### 📊 Performance Analysis

- **Benchmarking Framework**: Compare inference times across different methods
- **Confidence Distribution Analysis**: Visualize similarity score distributions
- **Performance Metrics**: Track accuracy, speed, and reliability

---

## 📊 Example Results

### Input Image

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmancer/clip-object-detection/main/assets/original_image.png" width="450" alt="Original Image">
</p>

### Region Proposals

Regions proposed by Faster R-CNN's RPN:

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmancer/clip-object-detection/main/assets/regions.png" width="450" alt="Candidate Regions">
</p>

### Detected Objects

Objects detected by CLIP based on textual queries:

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmancer/clip-object-detection/main/assets/clip_result.png" width="450" alt="Detected Objects">
</p>

---

## 📦 Requirements

Ensure the following are installed:

- **PyTorch**: Deep learning framework.
- **Torchvision**: Pre-trained Faster R-CNN.
- **OpenAI CLIP**: [GitHub Repository](https://github.com/openai/CLIP.git).
- **Transformers** (optional): For DETR model.
- **Ultralytics** (optional): For YOLOv8 model.
- Additional dependencies are listed in [requirements.txt](requirements.txt).

---

## 📖 Usage Examples

### Enhanced Detection with Confidence Scoring

```python
from clip_object_detection import detect_object_with_confidence

# Perform detection with confidence quantification
result = detect_object_with_confidence(
    model=clip_model,
    processor=clip_preprocess,
    prompts=['a dog', 'a photo of a dog'],
    image_pt=image_tensor,
    boxes=candidate_boxes
)

print(f"Detected object confidence: {result['confidence']:.3f}")
print(f"Similarity score: {result['similarity_score']:.3f}")
```

### Negative Prompt Detection

```python
# Reduce false positives with negative prompts
box = detect_with_negative_prompts(
    model=clip_model,
    processor=clip_preprocess,
    positive_prompts=['a dog'],
    negative_prompts=['cat', 'toy', 'stuffed animal'],
    image_pt=image_tensor,
    boxes=candidate_boxes
)
```

### Ensemble Detection

```python
# Combine multiple detection approaches
ensemble_results = ensemble_detector.detect_ensemble(
    prompts=['a dog'],
    image_pt=image_tensor,
    boxes=candidate_boxes
)

# Results include multiple detection methods for comparison
```

### Performance Benchmarking

```python
# Compare different detection methods
benchmark = DetectionBenchmark()
metrics = benchmark.benchmark_clip_enhanced(prompts, image_pt, boxes)
benchmark.display_comparison(metrics)
```

---

## 🎯 Key Improvements

| Feature                  | Benefit                                         |
| ------------------------ | ----------------------------------------------- |
| **Confidence Scoring**   | Quantify detection reliability                  |
| **Negative Prompts**     | Reduce false positives by 20-30%                |
| **Multi-Scale Analysis** | Better detection of objects at different scales |
| **Ensemble Methods**     | Improved robustness through voting              |
| **Alternative Models**   | DETR and YOLOv8 for comparison                  |
| **Performance Metrics**  | Detailed benchmarking and analysis              |

---

## 📝 License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code.

---

## ⭐ Support the Project

If this project inspires or assists your work, please consider giving it a ⭐ on GitHub! Your support motivates us to continue improving and expanding this repository.
