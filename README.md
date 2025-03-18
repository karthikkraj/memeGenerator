
# Meme Generator using GAN and CLIP

A cutting-edge **Meme Generator** project that combines **Generative Adversarial Networks (GANs)** and **CLIP (Contrastive Language–Image Pretraining)** to generate memes based on user-provided captions. The GAN generates realistic images, while CLIP ensures the generated image semantically aligns with the input text.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [Folder Structure](#folder-structure)
- [Setup & Prerequisites](#setup--prerequisites)
- [How to Use](#how-to-use)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🚀 Project Overview

This project utilizes the power of GANs to create meme images and leverages CLIP's text-image understanding capabilities to ensure the generated meme matches the provided caption. It demonstrates advanced concepts in **deep learning**, **natural language processing**, and **computer vision**.

---

## 🛠️ Model Architecture

1. **GAN Model**: Generates meme images.
2. **CLIP Model**: Matches generated images with input captions to ensure semantic relevance.

---

## 💻 Technologies Used

- **Python 3**
- **PyTorch**: For building and training GAN & CLIP models.
- **CLIP Model**: Pretrained language-image model from OpenAI.
- **NumPy, Pandas**: Data processing.

---

## 📂 Folder Structure

```
├── 2000_testdata.csv       # Dataset file
├── clip_model.py           # CLIP model integration code
├── data_loader.py          # Data loading and preprocessing
├── gan_model.py            # GAN model definition
├── generate_meme.py        # Meme generation script
├── train.py                # Training script
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

---

## 📥 Setup & Prerequisites

1. Python 3.8+ installed.
2. Install required Python packages:

```bash
pip install torch torchvision pandas numpy
```

3. Download CLIP pre-trained weights (if not auto-downloaded).

---

## ▶️ How to Use

1. **Clone the Repository**

```bash
git clone https://github.com/karthikkraj/memeGenerator.git
cd memeGenerator
```

2. **Prepare Dataset:** Ensure `2000_testdata.csv` is in the working directory.

3. **Train the Model:**

```bash
python train.py
```

4. **Generate Memes:**

```bash
python generate_meme.py --caption "Your funny caption here"
```

Generated memes will be saved in the output folder.

---

## 🌟 Future Improvements

- Fine-tune CLIP model for better meme-caption alignment.
- Add GUI/web interface for real-time meme generation.
- Expand dataset for more diverse meme styles.
- Optimize GAN architecture for faster training.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📢 Author

**Karthik Raj**  
GitHub: [karthikkraj](https://github.com/karthikkraj)
 
**SriHarsha Bodicherla**  
GitHub: [jacktherizzler](https://github.com/jacktherizzler)
 
