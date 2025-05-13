# Taylor expansion-based Kolmogorov-Arnold Network for Blind Image Quality Assessment
<div align="center">
  
</div>

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

### Run TaylorKAN:
```bash
cd mid-level/Taylor_ablation
python train_taylorkan.py
```

### Run KANs:
```bash
cd mid-level/KANs
python train_kans.py
```

## Experimental results

### 15 mid-level features

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="3" align="center">BID2011</th>
      <th colspan="3" align="center">CID2013</th>
      <th colspan="3" align="center">CLIVE</th>
      <th colspan="3" align="center">KonIQ-10k</th>
    </tr>
    <tr>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>KRCC</th>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>KRCC</th>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>KRCC</th>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>KRCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SVR</td>
      <td>0.619</td>
      <td>0.617</td>
      <td>0.461</td>
      <td>0.834</td>
      <td>0.810</td>
      <td>0.635</td>
      <td>0.630</td>
      <td>0.592</td>
      <td>0.409</td>
      <td>0.746</td>
      <td>0.691</td>
      <td>0.503</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>0.744</td>
      <td>0.729</td>
      <td>0.528</td>
      <td>0.808</td>
      <td>0.791</td>
      <td>0.588</td>
      <td>0.649</td>
      <td>0.552</td>
      <td>0.389</td>
      <td>0.753</td>
      <td>0.682</td>
      <td>0.494</td>
    </tr>
    <tr>
      <td>BsplineKAN</td>
      <td><strong>0.784</strong></td>
      <td>0.795</td>
      <td><strong>0.588</strong></td>
      <td>0.846</td>
      <td>0.843</td>
      <td>0.657</td>
      <td>0.599</td>
      <td>0.478</td>
      <td>0.328</td>
      <td>0.752</td>
      <td>0.680</td>
      <td>0.493</td>
    </tr>
    <tr>
      <td>EfficientKAN</td>
      <td>0.762</td>
      <td>0.785</td>
      <td>0.583</td>
      <td>0.862</td>
      <td>0.834</td>
      <td>0.648</td>
      <td>0.588</td>
      <td>0.505</td>
      <td>0.358</td>
      <td>0.753</td>
      <td>0.688</td>
      <td>0.499</td>
    </tr>
    <tr>
      <td>FourierKAN</td>
      <td>0.274</td>
      <td>0.344</td>
      <td>0.241</td>
      <td>0.498</td>
      <td>0.500</td>
      <td>0.330</td>
      <td>0.422</td>
      <td>0.412</td>
      <td>0.275</td>
      <td>0.404</td>
      <td>0.328</td>
      <td>0.222</td>
    </tr>
    <tr>
      <td>ChebyKAN</td>
      <td>0.700</td>
      <td>0.703</td>
      <td>0.528</td>
      <td>0.808</td>
      <td>0.826</td>
      <td>0.632</td>
      <td>0.570</td>
      <td>0.447</td>
      <td>0.312</td>
      <td>0.749</td>
      <td>0.680</td>
      <td>0.491</td>
    </tr>
    <tr>
      <td>FastKAN</td>
      <td>0.695</td>
      <td>0.683</td>
      <td>0.491</td>
      <td>0.836</td>
      <td>0.781</td>
      <td>0.597</td>
      <td>0.564</td>
      <td>0.502</td>
      <td>0.356</td>
      <td>0.727</td>
      <td>0.649</td>
      <td>0.466</td>
    </tr>
    <tr>
      <td>BSRBF KAN</td>
      <td>0.675</td>
      <td>0.680</td>
      <td>0.496</td>
      <td>0.845</td>
      <td>0.795</td>
      <td>0.614</td>
      <td>0.562</td>
      <td>0.479</td>
      <td>0.334</td>
      <td>0.725</td>
      <td>0.650</td>
      <td>0.465</td>
    </tr>
    <tr>
      <td>HermiteKAN</td>
      <td>0.651</td>
      <td>0.740</td>
      <td>0.533</td>
      <td>0.825</td>
      <td><strong>0.845</strong></td>
      <td>0.661</td>
      <td>0.566</td>
      <td>0.502</td>
      <td>0.354</td>
      <td>0.754</td>
      <td>0.671</td>
      <td>0.484</td>
    </tr>
    <tr>
      <td>JacobiKAN</td>
      <td>0.709</td>
      <td><strong>0.789</strong></td>
      <td>0.580</td>
      <td>0.808</td>
      <td>0.775</td>
      <td>0.585</td>
      <td>0.545</td>
      <td>0.519</td>
      <td>0.365</td>
      <td>0.753</td>
      <td>0.689</td>
      <td>0.500</td>
    </tr>
    <tr>
      <td>WavKAN</td>
      <td>0.715</td>
      <td>0.730</td>
      <td>0.532</td>
      <td>0.827</td>
      <td>0.827</td>
      <td>0.641</td>
      <td>0.559</td>
      <td>0.482</td>
      <td>0.336</td>
      <td>0.759</td>
      <td>0.685</td>
      <td>0.497</td>
    </tr>
    <tr>
      <td>TaylorKAN (ours)</td>
      <td>0.756</td>
      <td>0.782</td>
      <td>0.578</td>
      <td><strong>0.871</strong></td>
      <td>0.851</td>
      <td><strong>0.666</strong></td>
      <td><strong>0.668</strong></td>
      <td><strong>0.582</strong></td>
      <td><strong>0.409</strong></td>
      <td><strong>0.766</strong></td>
      <td><strong>0.699</strong></td>
      <td><strong>0.509</strong></td>
    </tr>
  </tbody>
</table>

### 2048 high-level features

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="3" align="center">BID2011</th>
      <th colspan="3" align="center">CID2013</th>
      <th colspan="3" align="center">CLIVE</th>
      <th colspan="3" align="center">KonIQ-10k</th>
    </tr>
    <tr>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>KRCC</th>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>KRCC</th>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>KRCC</th>
      <th>PLCC</th>
      <th>SRCC</th>
      <th>KRCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SVR</td>
      <td>0.786</td>
      <td>0.782</td>
      <td>0.584</td>
      <td><strong>0.860</strong></td>
      <td><strong>0.882</strong></td>
      <td><strong>0.698</strong></td>
      <td>0.751</td>
      <td><strong>0.712</strong></td>
      <td><strong>0.527</strong></td>
      <td>0.839</td>
      <td>0.800</td>
      <td>0.609</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>0.750</td>
      <td>0.780</td>
      <td>0.585</td>
      <td>0.796</td>
      <td>0.825</td>
      <td>0.623</td>
      <td>0.637</td>
      <td>0.554</td>
      <td>0.410</td>
      <td>0.808</td>
      <td>0.763</td>
      <td>0.574</td>
    </tr>
    <tr>
      <td>BsplineKAN</td>
      <td>0.801</td>
      <td>0.805</td>
      <td>0.601</td>
      <td>0.849</td>
      <td>0.843</td>
      <td>0.673</td>
      <td>0.718</td>
      <td>0.660</td>
      <td>0.486</td>
      <td>0.828</td>
      <td>0.795</td>
      <td>0.601</td>
    </tr>
    <tr>
      <td>EfficientKAN</td>
      <td>0.713</td>
      <td>0.745</td>
      <td>0.573</td>
      <td>0.740</td>
      <td>0.791</td>
      <td>0.599</td>
      <td>0.596</td>
      <td>0.536</td>
      <td>0.387</td>
      <td>0.815</td>
      <td>0.785</td>
      <td>0.593</td>
    </tr>
    <tr>
      <td>FourierKAN</td>
      <td>0.337</td>
      <td>0.358</td>
      <td>0.233</td>
      <td>0.265</td>
      <td>0.250</td>
      <td>0.175</td>
      <td>0.052</td>
      <td>0.054</td>
      <td>0.038</td>
      <td>0.096</td>
      <td>0.092</td>
      <td>0.062</td>
    </tr>
    <tr>
      <td>ChebyKAN</td>
      <td>0.821</td>
      <td>0.812</td>
      <td><strong>0.611</strong></td>
      <td>0.630</td>
      <td>0.665</td>
      <td>0.484</td>
      <td>0.662</td>
      <td>0.587</td>
      <td>0.413</td>
      <td>0.824</td>
      <td>0.790</td>
      <td>0.597</td>
    </tr>
    <tr>
      <td>FastKAN</td>
      <td>0.787</td>
      <td>0.779</td>
      <td>0.574</td>
      <td>0.841</td>
      <td>0.865</td>
      <td>0.680</td>
      <td>0.730</td>
      <td>0.643</td>
      <td>0.469</td>
      <td>0.845</td>
      <td><strong>0.813</strong></td>
      <td><strong>0.622</strong></td>
    </tr>
    <tr>
      <td>BSRBF KAN</td>
      <td>0.811</td>
      <td>0.795</td>
      <td>0.591</td>
      <td>0.828</td>
      <td>0.820</td>
      <td>0.657</td>
      <td>0.733</td>
      <td>0.649</td>
      <td>0.477</td>
      <td>0.841</td>
      <td>0.809</td>
      <td>0.616</td>
    </tr>
    <tr>
      <td>HermiteKAN</td>
      <td><strong>0.822</strong></td>
      <td><strong>0.814</strong></td>
      <td>0.610</td>
      <td>0.604</td>
      <td>0.687</td>
      <td>0.505</td>
      <td>0.670</td>
      <td>0.647</td>
      <td>0.465</td>
      <td>0.839</td>
      <td>0.804</td>
      <td>0.609</td>
    </tr>
    <tr>
      <td>JacobiKAN</td>
      <td>0.820</td>
      <td>0.806</td>
      <td>0.603</td>
      <td>0.596</td>
      <td>0.605</td>
      <td>0.446</td>
      <td>0.733</td>
      <td>0.651</td>
      <td>0.478</td>
      <td>0.842</td>
      <td>0.803</td>
      <td>0.611</td>
    </tr>
    <tr>
      <td>WavKAN</td>
      <td>0.767</td>
      <td>0.735</td>
      <td>0.531</td>
      <td>0.844</td>
      <td>0.856</td>
      <td>0.661</td>
      <td><strong>0.752</strong></td>
      <td>0.676</td>
      <td>0.494</td>
      <td>0.810</td>
      <td>0.777</td>
      <td>0.583</td>
    </tr>
    <tr>
      <td>TaylorKAN (ours)</td>
      <td>0.797</td>
      <td>0.813</td>
      <td><strong>0.611</strong></td>
      <td>0.788</td>
      <td>0.780</td>
      <td>0.595</td>
      <td>0.696</td>
      <td>0.598</td>
      <td>0.446</td>
      <td><strong>0.850</strong></td>
      <td>0.811</td>
      <td>0.621</td>
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
