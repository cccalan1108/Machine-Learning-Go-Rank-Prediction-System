# 圍棋模型訓練與預測專案

## 專案概述
本專案實作了一個基於多模態（序列 + 表格特徵）的 Stacking Ensemble Model，整合多種機器學習子模型，進行訓練、驗證與最終預測輸出。

## 專案結構
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


## 檔案說明

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





