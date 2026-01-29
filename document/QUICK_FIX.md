# ⚡ クイックフィックス：インデックス重複エラー

**エラー**: `[G1違反] 訓練とテストが418行重複しています`

---

## 🔧 修正方法（1分で完了）

### 方法1: run_titanic.pyを修正（推奨）

`run_titanic.py`の7-9行目の後に、以下を追加：

```python
import pandas as pd
from kaggle_alloy_implementation import KagglePipeline

def main():
    # データ読み込み
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # 👇 ここに追加 👇
    test_passenger_ids = test_df['PassengerId'].copy()
    train_df = train_df.set_index('PassengerId')
    test_df = test_df.set_index('PassengerId')
    # 👆 ここまで追加 👆
    
    # パイプライン実行
    pipeline = KagglePipeline()
    predictions, results = pipeline.execute(train_df, test_df)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'PassengerId': test_passenger_ids,  # 👈 変更
        'Survived': predictions
    })
    submission.to_csv('submission.csv', index=False)
    
    print("\n✅ submission.csv作成完了！")
    return submission, results
```

### 方法2: 修正済みファイルを使用

`run_titanic_fixed.py`をダウンロードして、`run_titanic.py`にリネーム。

---

## ▶️ 実行

```bash
python run_titanic.py
```

## ✅ 期待される出力

```
✅ インデックス設定完了
  訓練: 1 - 891
  テスト: 892 - 1309
  重複チェック: 0個

✅ [G1] 訓練・テスト分離チェック合格
✅ [G2] 特徴量エンジニアリング完了
✅ [G3] モデル構築完了
✅ すべてのAlloy制約（G1-G3）を満たしました！

📊 訓練精度: 0.8XXX
📊 CV精度: 0.7XXX
✅ submission.csv作成完了！
```

---

## 🎯 なぜこのエラーが起きたのか

**問題**：
```python
train_df = pd.read_csv('train.csv')  # インデックス: 0-890
test_df = pd.read_csv('test.csv')    # インデックス: 0-417
# → 418個重複！
```

**解決**：
```python
train_df.set_index('PassengerId')  # インデックス: 1-891
test_df.set_index('PassengerId')   # インデックス: 892-1309
# → 重複なし！
```

**Alloy制約との関係**：
```alloy
fact TrainTestSeparation {
  all ds: Dataset |
    no ds.train.rows & ds.test.rows
}
```
→ この制約が**早期発見**に貢献した！

---

## 📝 完全版スクリプト

完全な修正版は`run_titanic_fixed.py`を参照してください。

---

**作成日**: 2026年1月28日  
**修正時間**: 1分
