# Taylor expansion-based Kolmogorov-Arnold Network for Blind Image Quality Assessment
[![arXiv](https://img.shields.io/badge/arXiv-2409.07762-B31B1B.svg)](https://arxiv.org/abs/2409.07762)


![TaylorKAN](https://github.com/CUC-Chen/KAN4IQA/raw/main/assets/taylorkan.png)

## 🔥 News

- Dec 21, 2024: 👋 Our paper is accepted to ICASSP 2025!
- Oct 28, 2024: 👋 We release our code.
- Sep 12, 2024: 👋 We release our paper on arXiv.


## Environment setup

```bash
# 1. Clone the repository
git clone https://github.com/CUC-Chen/KAN4IQA.git
cd KAN4IQA

# 2. (Optional) Create and activate Conda environment
conda create -n kan4iqa python=3.10 -y
conda activate kan4iqa

# 3. Install dependencies
pip install -r requirements.txt
```

## Train and evaluation
```bash
cd scripts
python train.py
```


## Results

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2" align="center">BID</th>
      <th colspan="2" align="center">CLIVE</th>
      <th colspan="2" align="center">KonIQ</th>
      <th colspan="2" align="center">SPAQ</th>
      <th colspan="2" align="center">FLIVE</th>
    </tr>
    <tr>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>PLCC</th>
      <th>SRCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SVR</td>
      <td><u>0.834</u></td>
      <td><u>0.842</u></td>
      <td><u>0.754</u></td>
      <td>0.691</td>
      <td><strong>0.862</strong></td>
      <td><strong>0.850</strong></td>
      <td>0.834</td>
      <td>0.842</td>
      <td><u>0.537</u></td>
      <td><strong>0.510</strong></td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>0.750</td>
      <td>0.780</td>
      <td>0.637</td>
      <td>0.554</td>
      <td>0.808</td>
      <td>0.763</td>
      <td><u>0.855</u></td>
      <td><u>0.860</u></td>
      <td>0.370</td>
      <td>0.319</td>
    </tr>
    <tr>
      <td>KAN</td>
      <td>0.746</td>
      <td>0.756</td>
      <td>0.720</td>
      <td>0.681</td>
      <td>0.772</td>
      <td>0.739</td>
      <td>0.812</td>
      <td>0.810</td>
      <td>0.419</td>
      <td>0.376</td>
    </tr>
    <tr>
      <td>FastKAN</td>
      <td>0.813</td>
      <td>0.794</td>
      <td>0.731</td>
      <td><u>0.699</u></td>
      <td>0.805</td>
      <td>0.778</td>
      <td>0.845</td>
      <td>0.841</td>
      <td>0.504</td>
      <td>0.413</td>
    </tr>
    <tr>
      <td>ChebyKAN</td>
      <td>0.751</td>
      <td>0.736</td>
      <td>0.679</td>
      <td>0.549</td>
      <td>0.749</td>
      <td>0.716</td>
      <td>0.811</td>
      <td>0.807</td>
      <td>0.484</td>
      <td>0.344</td>
    </tr>
    <tr>
      <td>JacobiKAN</td>
      <td>0.762</td>
      <td>0.762</td>
      <td>0.721</td>
      <td>0.628</td>
      <td>0.782</td>
      <td>0.751</td>
      <td>0.792</td>
      <td>0.784</td>
      <td>0.495</td>
      <td>0.407</td>
    </tr>
    <tr>
      <td>WavKAN</td>
      <td>0.767</td>
      <td>0.735</td>
      <td>0.752</td>
      <td>0.676</td>
      <td>0.810</td>
      <td>0.777</td>
      <td>0.792</td>
      <td>0.784</td>
      <td>-0.006</td>
      <td>0.010</td>
    </tr>
    <tr>
      <td>HermiteKAN</td>
      <td>0.696</td>
      <td>0.656</td>
      <td>0.614</td>
      <td>0.575</td>
      <td>0.737</td>
      <td>0.702</td>
      <td>0.802</td>
      <td>0.800</td>
      <td>0.447</td>
      <td>0.369</td>
    </tr>
    <tr>
      <td>BSRBFKAN</td>
      <td>0.793</td>
      <td>0.763</td>
      <td>0.737</td>
      <td>0.692</td>
      <td>0.817</td>
      <td>0.792</td>
      <td>0.846</td>
      <td>0.841</td>
      <td>0.517</td>
      <td>0.430</td>
    </tr>
    <tr>
      <td>FourierKAN</td>
      <td>0.337</td>
      <td>0.358</td>
      <td>0.052</td>
      <td>0.054</td>
      <td>0.096</td>
      <td>0.092</td>
      <td>0.314</td>
      <td>0.274</td>
      <td>-0.049</td>
      <td>-0.028</td>
    </tr>
    <tr>
      <td>EfficientKAN</td>
      <td>0.657</td>
      <td>0.694</td>
      <td>0.731</td>
      <td>0.684</td>
      <td>0.779</td>
      <td>0.752</td>
      <td>0.742</td>
      <td>0.748</td>
      <td>0.473</td>
      <td>0.424</td>
    </tr>
    <tr>
      <td>TaylorKAN (ours)</td>
      <td><strong>0.842</strong></td>
      <td><strong>0.843</strong></td>
      <td><strong>0.783</strong></td>
      <td><strong>0.753</strong></td>
      <td><u>0.830</u></td>
      <td><u>0.816</u></td>
      <td><strong>0.856</strong></td>
      <td><strong>0.862</strong></td>
      <td><strong>0.539</strong></td>
      <td><u>0.484</u></td>
    </tr>
  </tbody>
</table>



## 📚 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{yu2025exploring,
  title={Exploring Kolmogorov-Arnold networks for realistic image sharpness assessment},
  author={Yu, Shaode and Chen, Ze and Yang, Zhimu and Gu, Jiacheng and Feng, Bizu and Sun, Qiurui},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
