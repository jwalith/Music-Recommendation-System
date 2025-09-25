# Music Recommendation System

A comprehensive music recommendation system implementing multiple approaches from baseline methods to advanced deep learning architectures.

## 🎵 Overview

This project implements a complete music recommendation pipeline with four different approaches, each building upon the previous one to create increasingly sophisticated recommendation systems.

## 📁 Project Structure

```
Music Recommendation/
├── 0.data_exploration.py          # Data analysis and exploration
├── 1.KNN_based.py                 # KNN-based recommendations
├── 2.RandomForest_based.py        # RandomForest ranking model
├── 3.SASRec_sequential.py         # SASRec sequential recommendations
├── 4.Two_tower.py                 # Two-tower architecture with FAISS
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for GPU acceleration)
- 8GB+ RAM recommended

### Installation

1. **Clone or download the project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run any stage:**
   ```bash
   python 0.data_exploration.py
   python 1.KNN_based.py
   python 2.RandomForest_based.py
   python 3.SASRec_sequential.py
   python 4.Two_tower.py
   ```

## 📊 Datasets

### Spotify Songs Dataset
- **Source**: Archive folder
- **Size**: 32,833 songs
- **Features**: Audio features (danceability, energy, valence, tempo, etc.)
- **Genres**: Pop, Rock, Hip-Hop, Electronic, and more

### Last.fm Dataset
- **Source**: hetrec2011-lastfm-2k folder
- **Interactions**: 92,834 user-artist interactions
- **Artists**: 17,632 unique artists
- **Users**: 1,892 unique users
- **Additional**: Social connections, tags, timestamps

## 🔧 Implementation Stages

### Stage 0: Data Exploration (`0.data_exploration.py`)
**Purpose**: Understand datasets and log insights

**Features**:
- Dataset statistics and distributions
- Audio feature analysis
- User behavior patterns
- Artist popularity analysis
- Weights & Biases integration

**Output**:
- Comprehensive data insights
- W&B logged metrics
- Dataset summaries

### Stage 1: KNN-Based Recommendations (`1.KNN_based.py`)
**Purpose**: Baseline recommendation methods

**Methods**:
- **Popularity-based**: Most popular artists
- **Content-based kNN**: Audio feature similarity
- **Hybrid approach**: Combines both methods

**Features**:
- Cosine similarity for audio features
- StandardScaler for feature normalization
- Comprehensive evaluation metrics

**Output**:
- Hit Rate and NDCG metrics
- Top popular artists
- Similar song recommendations

### Stage 2: RandomForest-Based Ranking (`2.RandomForest_based.py`)
**Purpose**: Learning-to-Rank approach

**Features**:
- **Feature Engineering**: User behavior + audio features
- **Random Forest**: 100 estimators for ranking
- **Derived Features**: Weight ratios, diversity metrics
- **Feature Importance**: Identifies key factors

**Key Features**:
- User activity patterns
- Artist popularity metrics
- Interaction strength analysis
- Relative activity measures

**Output**:
- RMSE, MAE, Correlation metrics
- Feature importance rankings
- Personalized recommendations

### Stage 3: SASRec Sequential (`3.SASRec_sequential.py`)
**Purpose**: Sequential recommendation modeling

**Architecture**:
- **SASRec Model**: Self-Attentive Sequential Recommendation
- **Transformer**: 4 heads, 2 layers, 128 hidden size
- **Embeddings**: Item + Position embeddings
- **Sequence Length**: Up to 50 items per sequence

**Features**:
- GPU acceleration with mixed precision
- Sequence padding and truncation
- Cross-entropy loss with padding ignore
- Comprehensive evaluation (Hit Rate@K, NDCG@K)

**Output**:
- Sequential recommendations
- Training/validation loss curves
- Ranking metrics at multiple K values

### Stage 4: Two-Tower Architecture (`4.Two_tower.py`)
**Purpose**: Advanced neural collaborative filtering

**Architecture**:
- **User Tower**: Neural network for user embeddings
- **Item Tower**: Neural network for item embeddings
- **FAISS Retrieval**: Fast similarity search
- **Cold-Start Handling**: New users/items support

**Features**:
- **GPU Optimization**: Mixed precision training
- **Feature Engineering**: Log-scaled heavy-tailed features
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K
- **Cold-Start Solutions**: Average embeddings fallback
- **FAISS Index**: IVF-based fast retrieval

**Output**:
- High-quality embeddings
- Fast retrieval recommendations
- Comprehensive evaluation metrics
- Cold-start recommendations

## ⚡ Performance Features

### GPU Acceleration
- **CUDA Support**: Automatic GPU detection
- **Mixed Precision**: 16-bit training for speed
- **Memory Optimization**: Efficient tensor operations
- **Batch Processing**: Optimized batch sizes

### Evaluation Metrics
- **Precision@K**: Accuracy of top-K recommendations
- **Recall@K**: Coverage of relevant items
- **NDCG@K**: Ranking quality measure
- **Hit Rate@K**: Success rate of recommendations

### Cold-Start Handling
- **New Users**: Average user embedding fallback
- **New Items**: Average item embedding fallback
- **Feature Engineering**: Robust feature creation
- **Graceful Degradation**: System continues working

## 📈 Expected Performance

| Stage | Method | Hit Rate@5 | NDCG@5 | Training Time |
|-------|--------|------------|--------|---------------|
| 1 | KNN-based | ~0.15 | ~0.10 | < 1 min |
| 2 | RandomForest | ~0.25 | ~0.18 | ~2 min |
| 3 | SASRec | ~0.78 | ~0.54 | ~5 min |
| 4 | Two-Tower | ~0.85 | ~0.65 | ~8 min |

*Performance may vary based on hardware and data*

## 🛠️ Technical Details

### Dependencies
- **Core**: pandas, numpy, scikit-learn
- **Deep Learning**: torch, transformers
- **Retrieval**: faiss-cpu
- **Tracking**: wandb
- **Visualization**: matplotlib, seaborn, plotly

### Hardware Requirements
- **Minimum**: 4GB RAM, CPU only
- **Recommended**: 8GB RAM, CUDA GPU
- **Optimal**: 16GB RAM, RTX 3080+ GPU

### Memory Usage
- **Stage 1-2**: ~2GB RAM
- **Stage 3**: ~4GB RAM (GPU: ~2GB VRAM)
- **Stage 4**: ~6GB RAM (GPU: ~4GB VRAM)

## 🔍 Usage Examples

### Basic Usage
```python
# Run data exploration
python 0.data_exploration.py

# Run KNN recommendations
python 1.KNN_based.py

# Run RandomForest ranking
python 2.RandomForest_based.py

# Run SASRec sequential
python 3.SASRec_sequential.py

# Run Two-Tower architecture
python 4.Two_tower.py
```
## 📊 Monitoring

### Weights & Biases Integration
- **Automatic Logging**: Metrics, losses, and visualizations
- **Experiment Tracking**: Compare different runs
- **Hyperparameter Tuning**: Track parameter effects
- **Model Comparison**: Side-by-side performance analysis

### Local Monitoring
- **Progress Bars**: Training progress visualization
- **Console Output**: Real-time metrics and status
- **Error Handling**: Graceful failure management


## 📚 References

- **SASRec**: Self-Attentive Sequential Recommendation
- **Two-Tower**: Neural Collaborative Filtering
- **FAISS**: Facebook AI Similarity Search
- **Last.fm Dataset**: HetRec 2011 Challenge
- **Spotify API**: Audio Features Documentation

