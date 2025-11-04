# Image Understanding & Retrieval System 🖼️🔍

A lightweight image understanding system that performs **multi-task learning** using Vision Transformers (DeiT-Tiny and MobileViT-XXS) for:
- **Image Classification** (10 object classes)
- **Attribute Prediction** (color, material, condition)
- **Text-based Image Retrieval** (search images using natural language)

## 🌟 Features

- **Dual Model Support**: Compare results from DeiT-Tiny and MobileViT-XXS
- **Multi-task Prediction**: Predict object class, color, material, and condition simultaneously
- **Text-based Search**: Find images using queries like "blue plastic bottle" or "black leather bag"
- **Auto-loading Gallery**: Automatically indexes images on startup
- **Model-specific Predictions**: Each model makes independent predictions, giving different search results
- **Interactive Web UI**: Built with Gradio for easy interaction

## 📋 Requirements

```bash
pip install torch torchvision timm pillow numpy scikit-learn gradio tqdm
```

## 🚀 Quick Start

### Prerequisites

Install required packages:
```bash
pip install torch torchvision timm pillow numpy scikit-learn gradio tqdm
```

### Setup Instructions

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. **Add your image dataset**:
   - Create an `images/` folder in the project root
   - Add your image dataset to this folder
   - The UI will automatically load and index all images on startup
   
   **Note**: The `images/` folder is excluded from version control due to size. You need to add your own images.
   
   **Expected structure**:
   ```
   project-root/
   ├── images/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ... (your images)
   ├── best_model1_deit_tiny.pth
   ├── best_model1_mobilevit_xxs.pth
   ├── image_understanding_ui.py
   └── ...
   ```

3. **Run the UI**:
```bash
python image_understanding_ui.py
```

4. **Access the interface**:
   - The browser will automatically open at `http://127.0.0.1:7860`
   - Wait for the gallery to load (progress shown in console)
   - Both models will analyze all images (takes ~2-3 minutes for 600 images)

## 🎯 Supported Categories

### Object Classes (10)
Backpack, Bag, Bottle, Cap, Jacket, Jug, Shoe, Sneaker, Sunglasses, Wallet

### Colors (15)
Beige, Black, Blue, Brown, Gold, Gray, Green, Orange, Pink, Purple, Red, Silver, White, Yellow, Multi

### Materials (9)
Canvas, Ceramic, Cotton, Fabric, Glass, Leather, Metal, Plastic, Synthetic

### Conditions (4)
Excellent, Good, Fair, Worn

## 💡 Usage Examples

### Image Classification
1. Go to "📸 Image Classification" tab
2. Upload an image
3. Select model (DeiT-Tiny or MobileViT-XXS)
4. Click "Predict" to see class and attributes

### Text-based Search
1. Go to "🔎 Text-based Image Search" tab
2. Enter a query (e.g., "red leather bag")
3. Select model to use for search
4. View matching images

**Tip**: Try the same query with both models to see how their different predictions affect results!

## 🏗️ Architecture

### Multi-task Vision Transformer
Both models use a shared backbone with task-specific heads:
- **Backbone**: DeiT-Tiny or MobileViT-XXS (frozen or fine-tuned)
- **Classification Head**: 10 classes
- **Color Head**: 15 colors
- **Material Head**: 9 materials
- **Condition Head**: 4 conditions

### Text-based Retrieval
- Each model predicts attributes for all images
- TF-IDF indexing on predicted attributes
- Cosine similarity for ranking results
- Model-specific search (different models → different results)

## 📊 Training Details

Models are trained using:
- **Loss**: Combined cross-entropy for all tasks
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR
- **Data Augmentation**: Random crops, flips, color jitter
- **Multi-task Learning**: Simultaneous training on all prediction tasks

See `milestone2_image_retrieval.ipynb` for full training code.

## 📁 Project Structure

```
.
├── image_understanding_ui.py          # Main UI application
├── milestone2_image_retrieval.ipynb   # Training & evaluation notebook
├── best_model1_deit_tiny.pth         # DeiT-Tiny trained weights
├── best_model1_mobilevit_xxs.pth     # MobileViT-XXS trained weights
├── images/                            # Image dataset folder
└── README.md                          # This file
```

## 🔧 Technical Notes

- **Device**: Automatically uses GPU if available, falls back to CPU
- **Image Size**: 224x224 pixels
- **Batch Processing**: Progress bars for gallery loading
- **Memory Efficient**: Stores features separately for each model
- **No External Dependencies**: No transformers library or external datasets needed for UI

## 📝 Citation

If you use this code, please cite:
```
Image Understanding & Retrieval System
Multi-task Vision Transformers for Image Classification and Attribute Prediction
```

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

This project is available for educational and research purposes.

---

**Note**: Model weights (`.pth` files) and images are excluded from version control due to size. You'll need to train the models using the provided notebook or use your own pre-trained weights.

