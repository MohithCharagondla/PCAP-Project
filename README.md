# CUDA Image Processing

## Overview
This project applies image processing techniques using CUDA for GPU acceleration. The following operations are implemented:
- **Blur**
- **Sharpen**
- **Outline Detection**
- **Color Inversion**
- **Black & White Conversion**

CUDA kernels are used to efficiently apply convolution-based filters and pixel-wise transformations.

---

## Operations & Kernel Choices

### 1️⃣ Blur (Box Filter)
- **Kernel Shape**: Square (Box Filter)
- **Kernel Size**: `3x3` to `21x21` (based on intensity)
- **Values**:
  ```
  Each element = 1 / (kernel size^2)
  ```
- **Why?**
  - Larger kernel sizes create stronger blurring effects.
  - This averaging filter smooths images by reducing noise and detail.

---

### 2️⃣ Sharpening
- **Kernel Shape**: Square
- **Kernel Size**: `3x3`
- **Values**:
  ```
  [  0   -intensity/4   0  ]
  [ -intensity/4  (1 + intensity)  -intensity/4 ]
  [  0   -intensity/4   0  ]
  ```
- **Why?**
  - Enhances edges by amplifying differences between pixels.
  - Normalized to prevent excessive brightness changes.

---

### 3️⃣ Outline (Edge Detection)
- **Kernel Shape**: Square
- **Kernel Size**: `3x3` (Fixed)
- **Values**:
  ```
  [ -1  -1  -1 ]
  [ -1   8  -1 ]
  [ -1  -1  -1 ]
  ```
- **Why?**
  - Detects edges by highlighting areas with high contrast.
  - The center weight (8) enhances edge visibility.

---

### 4️⃣ Color Inversion
- **Formula**:
  ```
  new_pixel = 255 - original_pixel
  ```
- **Why?**
  - Inverts colors for a negative effect.

---

### 5️⃣ Black & White Conversion
- **Formula**:
  ```
  gray = 0.299 R + 0.587 G + 0.114 B
  ```
- **Why?**
  - Uses standard grayscale conversion.
  - Preserves human-perceived brightness levels.

---

## Installation & Usage

### Requirements
- OpenCV
- CUDA
- C++ Compiler with CUDA support

### Compilation
```sh
nvcc image_filter.cu -o image_filter `pkg-config --cflags --libs opencv4`
```

### Run
```sh
./image_filter <image_path> <operation> <intensity>
```
Example:
```sh
./image_filter input.jpg 1 5  # Apply blur with intensity 5
```

### Output
The processed image is saved as `output.png`.

---

🚀 **Happy Coding!**

