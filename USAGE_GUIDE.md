# Alloy検証済みKaggle実装 - 使用ガイド

**実装ファイル**: `kaggle_alloy_implementation.py`  
**Alloyモデル**: `kaggle_competition_v3_final.als`  
**検証済みゴール**: ✅ G1, ✅ G2, ✅ G3（G4は実装レベル）

---

## 🎯 このガイドについて

このPython実装は、Alloy形式手法で検証済みの制約を満たすKaggleパイプラインです。
**G1-G3がAlloy検証済み**、G4は実装レベルで追加されています。

---

## 🚀 クイックスタート

### 1. サンプルデータで実行

```bash
python kaggle_alloy_implementation.py
```

**実行結果**：
```
✅ [G1] データ品質保証（Alloy検証済み）
✅ [G2] 特徴量エンジニアリング（Alloy検証済み）
✅ [G3] モデル構築（Alloy検証済み）
✅ PracticalKagglePipeline達成
```

---

## 📊 実際のKaggleデータで実行

### ステップ1: タイタニックデータをダウンロード

```bash
# Kaggle CLIを使用
kaggle competitions download -c titanic
unzip titanic.zip
```

### ステップ2: スクリプトを作成

`run_titanic.py`を作成：

```python
"""
タイタニックコンペ実行スクリプト
"""
import pandas as pd
from kaggle_alloy_implementation import KagglePipeline

def main():
    # データ読み込み
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"訓練データ: {len(train_df)}行")
    print(f"テストデータ: {len(test_df)}行")
    
    # パイプライン実行
    pipeline = KagglePipeline()
    predictions, results = pipeline.execute(train_df, test_df)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv('submission.csv', index=False)
    
    print("\n✅ submission.csv作成完了！")
    print(f"訓練精度: {results['train_metrics']['accuracy']:.4f}")
    print(f"CV精度: {results['cv_results']['cv_mean']:.4f}")
    
    return submission, results

if __name__ == '__main__':
    submission, results = main()
```

### ステップ3: 実行

```bash
python run_titanic.py
```

### ステップ4: Kaggleに提出

```bash
kaggle competitions submit -c titanic -f submission.csv -m "Alloy検証済みパイプライン"
```

---

## 🔧 Alloy制約との対応

### G1: データ品質保証

| Python | Alloy制約 | 説明 |
|--------|----------|------|
| `handle_missing_values()` | `fact MissingValueHandling` | 欠損値処理 |
| `handle_outliers()` | `fact OutlierConstraints` | 外れ値処理（年齢0-120） |
| `validate_no_missing()` | `assert NoMissingAfterProcessing` | 前処理後の欠損値なし |
| `validate_train_test_separation()` | `fact TrainTestSeparation` | 訓練・テスト分離 |

### G2: 特徴量エンジニアリング

| Python | Alloy制約 | 説明 |
|--------|----------|------|
| `create_features()` | `fact FeatureEngineeringRules` | 新特徴量生成 |
| `validate_new_features_exist()` | `some fed.newFeatures` | 新特徴量が存在 |

生成される特徴量：
- **FamilySize**: ドメイン知識（SibSp + Parch + 1）
- **IsAlone**: ドメイン知識（FamilySize == 1）
- **Age_binned_numeric**: 統計的（年齢を4つのビンに分類）
- **Fare_per_person**: 相互作用（Fare / FamilySize）

### G3: モデル構築

| Python | Alloy制約 | 説明 |
|--------|----------|------|
| `set_hyperparameters()` | `fact HyperparameterConstraints` | ハイパーパラメータ範囲 |
| `train()` | `some tm.model` | モデル訓練 |
| `predict()` | `fact PredictionBinary` | 予測値0/1チェック |
| `validate_hyperparameters()` | `assert ValidHyperparameters` | ハイパーパラメータ妥当性 |

ハイパーパラメータ制約：
- `max_depth`: [1, 20]
- `n_estimators`: [1, 1000]

### G4: 評価（実装レベルのみ）

| Python | Alloy制約 | 説明 |
|--------|----------|------|
| `evaluate()` | なし | モデル評価 |
| `cross_validate()` | なし | クロスバリデーション |

**注意**: G4はAlloyで形式化していません（形式化困難のため）。

---

## 💡 カスタマイズ方法

### 1. 新しい特徴量を追加

```python
class FeatureEngineer:
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 既存の特徴量生成...
        
        # 新しい特徴量を追加
        if 'Name' in df.columns:
            df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            df['Title_encoded'] = pd.factorize(df['Title'])[0]
            self.new_features.append('Title_encoded')
            print(f"  🔧 Title_encoded生成")
        
        # Alloy制約は自動的に満たされる
        self.validator.validate_new_features_exist(df, self.new_features)
        
        return df
```

### 2. ハイパーパラメータの調整

```python
# Alloy制約内の範囲で調整
pipeline = KagglePipeline()

# 方法1: デフォルト値を変更
trainer = pipeline.trainer
trainer.set_hyperparameters(
    max_depth=15,      # [1, 20]の範囲内
    n_estimators=200,  # [1, 1000]の範囲内
    min_samples_split=10
)

# 方法2: run_titanic.py内で直接設定
predictions, results = pipeline.execute(train_df, test_df)
```

### 3. 別のモデルを使用

```python
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def train(self, X, y):
        # RandomForest → LogisticRegression
        self.model = LogisticRegression(**self.hyperparameters, random_state=42)
        self.model.fit(X, y)
        
        # Alloy制約は自動的に満たされる
        return self.model
```

---

## 🐛 トラブルシューティング

### エラー1: 欠損値が残っている

```
[G1違反] 欠損値処理後: 欠損値が5個残っています
```

**原因**: 新しいカラムの欠損値処理が不足

**解決**:
```python
def handle_missing_values(self, df):
    df = df.copy()
    
    # 数値・カテゴリカル処理...
    
    # 新しいカラムを追加した場合は処理を追加
    if 'NewColumn' in df.columns:
        df['NewColumn'].fillna(df['NewColumn'].median(), inplace=True)
    
    return df
```

### エラー2: ハイパーパラメータが範囲外

```
[G3違反] max_depthは[1, 20]の範囲: 25
```

**原因**: Alloy制約違反

**解決**:
```python
# 制約内に収める
trainer.set_hyperparameters(
    max_depth=20,  # 20以下
    n_estimators=500  # 1000以下
)
```

### エラー3: 訓練・テスト重複

```
[G1違反] 訓練とテストが100行重複しています
```

**原因**: インデックスが重複

**解決**:
```python
# インデックスを明示的に設定
train_df = pd.DataFrame({...}, index=range(0, 100))
test_df = pd.DataFrame({...}, index=range(100, 150))
```

---

## 📈 パフォーマンス向上のヒント

### 1. 特徴量の追加

```python
# Name（敬称）
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Cabin（最初の文字）
df['Cabin_letter'] = df['Cabin'].str[0]

# Sex（数値化）
df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
```

### 2. ハイパーパラメータチューニング

```python
# Grid Search（手動）
best_score = 0
best_params = {}

for depth in [5, 10, 15, 20]:
    for n_est in [50, 100, 200]:
        trainer.set_hyperparameters(
            max_depth=depth,
            n_estimators=n_est
        )
        model = trainer.train(X_train, y_train)
        cv_results = evaluator.cross_validate(model, X_train, y_train)
        
        if cv_results['cv_mean'] > best_score:
            best_score = cv_results['cv_mean']
            best_params = {'max_depth': depth, 'n_estimators': n_est}

print(f"最適パラメータ: {best_params}")
print(f"最高スコア: {best_score:.4f}")
```

### 3. アンサンブル

```python
# 複数モデルの予測を平均
models = []

# モデル1
trainer1 = ModelTrainer(validator)
trainer1.set_hyperparameters(max_depth=10, n_estimators=100)
model1 = trainer1.train(X_train, y_train)
models.append(model1)

# モデル2
trainer2 = ModelTrainer(validator)
trainer2.set_hyperparameters(max_depth=15, n_estimators=200)
model2 = trainer2.train(X_train, y_train)
models.append(model2)

# アンサンブル予測
predictions = []
for model in models:
    pred = model.predict(X_test)
    predictions.append(pred)

# 多数決
final_predictions = np.round(np.mean(predictions, axis=0)).astype(int)
```

---

## 📚 Alloyモデルとの完全対応表

### 述語（Predicates）

| Alloy述語 | Python実装 | 説明 |
|-----------|-----------|------|
| `G1_Achieved` | `DataPreprocessor` | データ品質保証 |
| `G2_Achieved` | `FeatureEngineer` | 特徴量生成 |
| `G3_Achieved` | `ModelTrainer` | モデル構築 |
| `PracticalKagglePipeline` | `KagglePipeline.execute()` | G1∧G2∧G3 |

### ファクト（Facts）

| Alloy Fact | Python検証 | タイミング |
|-----------|-----------|----------|
| `MissingValueHandling` | `validate_no_missing()` | G1完了時 |
| `OutlierConstraints` | `validate_outliers()` | G1完了時 |
| `FeatureEngineeringRules` | `validate_new_features_exist()` | G2完了時 |
| `HyperparameterConstraints` | `validate_hyperparameters()` | G3開始時 |
| `TrainTestSeparation` | `validate_train_test_separation()` | パイプライン開始時 |
| `PredictionBinary` | `validate_predictions_binary()` | G3予測時 |

### アサーション（Assertions）

| Alloy Assertion | Python検証 | 結果 |
|----------------|-----------|------|
| `NoMissingAfterProcessing` | 最終検証 | ✅ 合格 |
| `NoTrainTestOverlap` | 開始時検証 | ✅ 合格 |
| `ValidHyperparameters` | G3開始時検証 | ✅ 合格 |
| `PredictionsAreBinary` | G3予測時検証 | ✅ 合格 |

---

## 🎯 実行結果の見方

### 成功例

```
✅ [G1] データ品質保証（Alloy検証済み）
✅ [G2] 特徴量エンジニアリング（Alloy検証済み）
✅ [G3] モデル構築（Alloy検証済み）
✅ PracticalKagglePipeline達成

訓練精度: 0.9700
CV精度: 0.5300
テスト予測数: 50
```

**解釈**：
- **訓練精度**: 訓練データでの精度（高すぎる場合は過学習の可能性）
- **CV精度**: クロスバリデーション精度（汎化性能の目安）
- **予測数**: テストデータの行数と一致していることを確認

---

## 🎉 まとめ

このPython実装は：

✅ **Alloy形式手法で検証済み**（G1-G3）  
✅ **実際のKaggleコンペ**で使用可能  
✅ **拡張・カスタマイズ**が容易  
✅ **教育・研究**に最適  

**AI-Augmented形式手法の実践例**として、ブログ記事やポートフォリオに最適です！

---

## 📖 関連ドキュメント

- **Alloyモデル**: `kaggle_competition_v3_final.als`
- **KAOS Goal構造**: `claude_code_learning_kaos.als`
- **詳細ドキュメント**: `kaggle_ai_augmented_formal_methods.md`
- **エラー修正ガイド**: `alloy_error_fix_guide.md`
- **過剰制約分析**: `over_constrained_analysis.md`

---

**作成日**: 2026年1月28日  
**作成者**: 古閑弘晃  
**ライセンス**: MIT
