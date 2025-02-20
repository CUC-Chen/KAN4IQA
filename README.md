# Taylor expansion-based Kolmogorov-Arnold Network for Blind Image Quality Assessment

<div align="center">
  
</div>

## 📌 Abstract

Score regression is important in blind image quality assessment (BIQA), with support vector regression (SVR) and multi-layer perceptrons (MLP) being widely applied. Recently, Kolmogorov-Arnold networks (KAN) have shown success in function approximation, yet their effectiveness in BIQA score regression remains largely unexplored. In this study, Taylor expansion-based KAN (TaylorKAN) is proposed which employs Taylor series as learnable activation functions on edges for representation learning. Then, 9 other KAN models are evaluated for BIQA alongside SVR, MLP and TaylorKAN, and 15 mid-level features and 2048 high-level features are prepared. On 4 image databases (BID, CID2013, CLIVE, and KonIQ-10k) with authentic distortions, TaylorKAN yields generally the best performance when utilizing mid-level features, whereas KANs perform worse than SVR when high-dimensional high-level features are used. Meanwhile, ablation studies exploring different combinations of modules and orders of Taylor series are conducted. The findings suggest that TaylorKAN holds promise for enhancing BIQA score regression, and further research should focus on addressing the issues of
high-dimensional features as well as its integration into deep networks for KAN-based no-reference image quality estimation.

### Main contributions

- TaylorKAN is proposed, with its parameters and the order of Taylor expansions determined for score prediction using 15 mid-level and 2048 high-level features.
- Twelve regression models, including SVR, MLP, and ten KAN models, are evaluated on four image databases containing authentic distortions.
- TaylorKAN delivers the best performance when mid-level features are used, while its limitation is identified in handling high-dimensional features.


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
