# ML Assignment 1: Go Rank Prediction System

This repository contains the implementation for Assignment 1 of the Machine Learning (CSIE5043) course, Fall 2025, at National Taiwan University. 
The goal of this assignment is to implement a Machine Learning model that can predict the playing strength (rank) of a Go player from their game records. The dataset contains Go games annotated with player rank labels. Your task is to build a model that uses extracted features from the games to accurately estimate the player’s rank.



# Kaggle 
https://www.kaggle.com/competitions/machine-learning-class-fall-2025-assignment-1-q-5/overview


# Dataset Description

### train_set：
The training set The training set consists of nine files, each corresponding to one rank (1D–9D). For example: log_9D_policy_train.txt contains data from 9-dan players, log_8D_policy_train.txt from 8-dan, and so on. Each file records all moves from many games of that rank. Each move includes multiple types of features (policy values, rank model outputs, strength score, lead information).

### test_set：
The test set The test set has the same format, but samples are aggregated from different game sets than the training data. Test samples are grouped several games per sample. This ensures that models cannot rely on memorization and must generalize to unseen data.

### sample_submission.csv：
Asample submission file in the correct format





# Project Structure

project/
│
├── trainml1.py          # Main training script (trains the StackX model)
├── Q5.py                # Testing and prediction script (generates submission.csv)
├── train_summary.json   # Training summary (cross‑validation results of each module)
├── model_stackx.pkl     # Trained final StackX model (pickle format)
├── submission.csv       # Final prediction output file
│
├── train_set/           # Training data folder (relative path)
└── test_set/            # Test data folder (relative path)



# File Descriptions


trainml.py – Model Training Script
The main training script is trainml1.py.
You can run it directly; the program will automatically load data folders via relative paths.
Usage:

```bash
python trainml.py --train_dir train_set --out_dir . --gpu --seq_len 120 --epochs 10
```

```bash
python trainml1.py \
  --train_dir ./train_set \
  --out_dir ./out_dir \
  --seq_len 120 \
  --epochs 10 \
  --batch_size 128 \
  --lr 0.007 \
  --seed 42 \
  --gpu
```


Argument Description

```bash
--train_dir: Training data directory (default: train_set)
--out_dir: Output directory (default: current directory)
--gpu: Use GPU acceleration
--seq_len: Sequence length (default: 120)
--epochs: Number of training epochs (default: 10)
--batch_size: Batch size (default: 128)
--lr: Learning rate (default: 0.007)
--seed: Random seed (default: 42)
```




# Q5.py


The main testing script is Q5.py.
It automatically loads model_stackx.pkl and generates the final prediction file under out_dir/.
Key Features:
- Pretrained stack loading: Loads the pre‑trained stacked model.
- Flexible ensembling: Supports multiple model ensembling strategies.
- Automated test handling: Automatically processes test files and generates predictions.
- Submission ready: Outputs a CSV submission file.


```bash
python Q5.py --test_dir test_set --model_path model_stackx.pkl --out_csv submission.csv
```

Argument Description

```bash
--test_dir: Test data directory (default: test_set)
--model_path: Path to the model file (default: model_stackx.pkl)
--out_csv: Output CSV file (default: submission.csv)
```



# Training Summary

Example train_summary.json:


```bash
{
  "oof_acc_seq_tf": 0.4015,
  "oof_acc_seq_bl": 0.3868,
  "oof_acc_tab": 0.2031,
  "oof_acc_meta": 0.4244
}
```

Meaning:
- seq_tf: Accuracy of the transformer‑based sequence model
- seq_bl: Accuracy of the baseline RNN model
- tab: Accuracy of the tabular‑feature model
- meta: Accuracy of the final stacked model (stacking meta model)





# Data Format

### Input Data Format

- Training data: log_XD_policy_train.txt, where X = 1–9 corresponds to different ranks.
- Test data: X.txt, where X is the file index.


### File Contents

Each line records a move and its corresponding feature information, including:
- Game ID: Game X
- Move: represented as B[coord] or W[coord] for black/white moves
- Feature vector:
  - Policy vector (9‑dim)
  - Value vector (9‑dim)
  - Rank probability vector (9‑dim)
  - Strength (1‑dim)
  - Winrate, Lead, Uncertainty (3‑dim)




# Model Architecture Overview

1. Sequence Models
- TinyTransformer: Transformer‑based sequence model capturing long‑term game dependencies.
- BiLSTM: Bidirectional LSTM model for time‑series information.
- Config: Total feature dimension is 79, with sequence length set to 120 moves.
2. Tabular Models
- Models: CatBoost or HistGradientBoostingClassifier (HGBT).
- Design: Extracts statistical information and game‑state dynamics from the entire game, and partitions features into opening, midgame, and endgame segments.
3. Fusion via Meta‑learning
- Meta‑learner: Logistic Regression.
- Fusion: Combines outputs from the sequence models and tabular models, along with additional side features, to form the final prediction.


# Feature Engineering Design
### Sequence Features (79‑dim)
- Basic features: Policy, Value, RankP (9‑dim each) + Strength, Winrate, Lead, Uncertainty
- Derived features: Max values and entropies (12‑dim total)
- Difference features: First‑ and second‑order differences (40‑dim total)
- Board position and color: Color indicators and normalized move positions


### Tabular Features
- Statistical features: mean, standard deviation, max/min, median, skewness, etc.
- Phase features: extracted separately for opening, midgame, and endgame.
- Preference features: style indicators such as entropy, maximum, and mode.


### Side Features (10‑dim)
- Entropies of Policy / Value / RankP
- Game length (log‑scaled)
- Standard deviation of Winrate
- Mean absolute Lead
- Mean Uncertainty
- Policy entropy in opening / midgame / endgame





# Training Pipeline
1.Data Parsing: Read game logs and convert them into model‑ready formats.
2.Feature Processing: Extract and integrate sequence and tabular features.
3.Model Training:
  - Train the Transformer and BiLSTM sequence models.
  - Train tabular models (CatBoost or HGBT).
4.Model Ensembling: Use Logistic Regression to combine outputs of multiple models into the final prediction.
5.Model Saving: Store the complete stacked model as model_stackx.pkl




# Prediction Pipeline
1.Model Loading: Load the trained model_stackx.pkl.
2.Data Processing: Parse test data and generate features consistent with the training phase.
3.Multi‑model Prediction:
  - Use Transformer and BiLSTM for sequence predictions.
  - Tabular models predict global features.
  - Incorporate side features for additional signals.
4.Model Ensembling: The meta‑learner aggregates predictions from all models.
5.Output Results: Generate the final submission file submission.csv (containing id and rank columns).




# Output Files
1.Training Phase
  - model_stackx.pkl: Complete stacked model
  - train_summary.json: Training summary information
2.Prediction Phase
  - submission.csv: Prediction results (with id and rank columns)





---------------------------------------------------------------------------------------------------------

# 圍棋模型訓練與預測專案

### 專案概述
本專案實作了一個基於多模態（序列 + 表格特徵）的 Stacking Ensemble Model，整合多種機器學習子模型，進行訓練、驗證與最終預測輸出。

### 專案結構
project/
│
├── trainml1.py            # 主訓練程式 (訓練StackX模型)
├── Q5.py                  # 測試與輸出預測結果 (生成submission.csv)
├── train_summary.json     # 訓練摘要（包含各模組的交叉驗證結果）
├── model_stackx.pkl       # 訓練完成的StackX最終模型（pickle格式）
├── submission.csv         # 最終預測輸出檔
│
├── train_set/             # 訓練資料夾（相對路徑）
└── test_set/              # 測試資料夾（相對路徑）


### 檔案說明

### trainml.py - 模型訓練腳本
訓練主程式位於 trainml1.py。
可直接執行，系統會使用相對路徑自動載入資料夾。

**使用方法：**
```bash
python trainml.py --train_dir train_set --out_dir . --gpu --seq_len 120 --epochs 10
```

```bash
python trainml1.py \
  --train_dir ./train_set \
  --out_dir ./out_dir \
  --seq_len 120 \
  --epochs 10 \
  --batch_size 128 \
  --lr 0.007 \
  --seed 42 \
  --gpu
```

**參數說明：**
- `--train_dir`: 訓練資料目錄（預設：train_set）
- `--out_dir`: 輸出目錄（預設：當前目錄）
- `--gpu`: 使用 GPU 加速
- `--seq_len`: 序列長度（預設：120）
- `--epochs`: 訓練輪數（預設：10）
- `--batch_size`: 批次大小（預設：128）
- `--lr`: 學習率（預設：0.007）
- `--seed`: 隨機種子（預設：42）

### Q5.py 
測試主程式為 Q5.py。
會自動載入 model_stackx.pkl 並在 out_dir/ 生成最終預測檔：

**主要特點：**
- 載入預訓練的模型堆疊
- 支援多種模型融合策略
- 自動處理測試檔案並生成預測結果
- 輸出 CSV 格式的提交檔案

**使用方法：**
```bash
python Q5.py --test_dir test_set --model_path model_stackx.pkl --out_csv submission.csv
```

**參數說明：**
- `--test_dir`: 測試資料目錄（預設：test_set）
- `--model_path`: 模型檔案路徑（預設：model_stackx.pkl）
- `--out_csv`: 輸出 CSV 檔案（預設：submission.csv）

### 訓練結果摘要
train_summary.json 範例：
{
  "oof_acc_seq_tf": 0.4015,
  "oof_acc_seq_bl": 0.3868,
  "oof_acc_tab": 0.2031,
  "oof_acc_meta": 0.4244
}
代表：
seq_tf: Transformer-based 序列模型準確率
seq_bl: Baseline RNN 模型準確率
tab: 表格特徵模型準確率
meta: 最終堆疊模型（Stacking Meta Model）準確率



## 資料格式

### 輸入資料格式
- **training data 訓練資料：** `log_XD_policy_train.txt` 其中 X = 1–9，對應不同段位。
- **test data 測試資料：** `X.txt` ，X 為檔案編號。

### 檔案內容
每一行記錄一個棋步與對應的特徵資訊，包含：
- 遊戲編號：`Game X`
- 棋步：以`B[座標]` 或 `W[座標]`表示黑白棋的落子位置
- 特徵向量：
  - Policy 向量（9維）
  - Value 向量（9維）
  - Rank probability 向量（9維）
  - Strength（1維）
  - Winrate, Lead, Uncertainty（3維）



## 模型架構概述

### 1. 序列模型(Sequence Models)
- **TinyTransformer**: 基於 Transformer 的序列模型，能捕捉棋局中的長期依存關係。
- **BiLSTM**: 雙向 LSTM 模型，擅長處理時間序列資訊。
- 特徵維度共 79 維，序列長度設定為 120 步。

### 2. Tabular 表格模型
- 採用 CatBoost 或 HistGradientBoostingClassifier (HGBT)。
- 從整場遊戲中萃取統計資訊與局勢變化，並依開局、中盤、終盤劃分特徵區段。

### 3. 融合 Meta-learning
- 以 Logistic Regression 作為元學習器 (meta-learner)。
- 結合序列模型與表格模型的輸出結果，同時納入額外側邊特徵，形成最終預測。



## 特徵工程設計

### 序列特徵（79維）
- 基礎特徵：Policy, Value, RankP (各9維) + Strength, Winrate, Lead, Uncertainty
- 衍生特徵：最大值與熵值（共 12 維）
- 差分特徵：包含一階與二階差分（共 40 維）
- 棋局位置與顏色：顏色標記與正規化棋步位置

### Tabular 特徵
- 統計特徵：平均值、標準差、最大/最小值、中位數、偏度等。
- 階段特徵：根據開局、中局、終局分段提取特徵。
- 偏好特徵：包括熵值、最大值與眾數等棋風指標。

### 側邊特徵（10維）
- Policy/Value/RankP 熵值
- 遊戲長度（取對數）
- Winrate 標準差
- Lead 絕對值均值
- Uncertainty 均值
- 開局/中局/終局 Policy 熵值

## 訓練流程

1. **資料解析**：讀取棋譜日誌並轉換為模型可用的格式。
2. **特徵處理**：進行序列與表格特徵的萃取與整合。
3. **模型訓練**：
   - 訓練 Transformer 與 BiLSTM 模型。
   - 訓練表格型模型（CatBoost 或 HGBT）。
4. **模型融合**：利用 Logistic Regression 將多模型輸出結合成最終預測。
5. **模型儲存**：保存完整堆疊模型於 model_stackx.pkl。

## 預測流程
1. **載入模型 **：讀取已訓練完成的 model_stackx.pkl
2. **資料處理 **：解析測試資料，生成與訓練階段一致的特徵格式。
3. **多模型預測 **：
   - 使用 Transformer 與 BiLSTM 進行序列預測。
   - 表格模型預測全局特徵。
   - 載入側邊特徵進行補強。
4. **模型融合 **：由 Meta-learner 整合各模型的預測結果。
5. **輸出結果 **：生成最終提交檔 submission.csv（包含 id 與 rank 欄位）。


## 輸出檔案

### 訓練階段
- `model_stackx.pkl`: 完整的模型堆疊
- `train_summary.json`: 訓練摘要資訊

### 預測階段
- `submission.csv`: 預測結果（包含 id 和 rank 欄位）













