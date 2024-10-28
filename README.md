# Taylor expansion based Kolmogorov-Arnold Network for Blind Image Quality Assessment

<div align="center">
  
</div>

## 📌 Abstract

Score prediction is important in image quality assessment, and support vector regression (SVR) and multi-perceptron layer (MLP) are widely used. Recently, Kolmogorov-Arnold networks (KAN) have been witnessed successful in function approximation and object classification, while little is known of their effectiveness on image quality prediction. This study presents Taylor expansion based KAN (TaylorKAN) that employs Taylor series as the learnable activation functions on edges for representation learning. And then, five popular KANs are explored for blind image quality estimation besides SVR, MLP and TaylorKAN. On four databases (BID2011, CID2013, CLIVE and KonIQ-10k) with diverse distortions, TaylorKAN achieves the best prediction when mid-level features are used, while the KANs cause inferior results compared to SVR when using high-dimensional deeply learned features. Ablation studies of different module combinations and different orders of Taylor expansion are also conducted. The KANs show promise in improving score prediction, and more attention should be paid to addressing high-dimensional features.

### Main contributions

- Image quality assessment is formulated as a feature selection and score prediction problem. 
- Learning-free and shallow-learning-based indicators perform as mid-level feature extractors, and consensus-based feature selection is proposed for informative feature screening.
- Fifteen indicators and four regression models are explored, and the hybrid indicator achieves considerable improvement on four realistic image datasets.


## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/CUC-Chen/KAN4IQA.git

# Navigate to the directory
cd KAN4IQA

# Install required packages
pip install -r requirements.txt
