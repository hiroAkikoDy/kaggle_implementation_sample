# Kaggleコンペティションへの AI-Augmented形式手法 適用例

**対象リポジトリ**: https://github.com/upura/python-kaggle-start-book  
**書籍**: 『PythonではじめるKaggleスタートブック』  
**例題**: タイタニック生存予測（典型的な二値分類問題）

---

## 📋 目次

1. [Kaggleワークフローの理解](#kaggleワークフローの理解)
2. [KAOS図によるゴール構造化](#kaos図によるゴール構造化)
3. [Alloy形式記法による制約記述](#alloy形式記法による制約記述)
4. [Claude Codeへの実装プロンプト](#claude-codeへの実装プロンプト)
5. [実装例とテスト](#実装例とテスト)
6. [効果の検証](#効果の検証)

---

## 1. Kaggleワークフローの理解

### 典型的なKaggleコンペティションのフロー

```
Phase 1: データ理解
  ↓
Phase 2: 探索的データ分析（EDA）
  ↓
Phase 3: データ前処理
  ↓
Phase 4: 特徴量エンジニアリング
  ↓
Phase 5: モデル訓練
  ↓
Phase 6: モデル評価
  ↓
Phase 7: 予測・提出
  ↓
Phase 8: 改善ループ
```

### 問題点：自然言語による曖昧性

**典型的な問題**：
```python
# 「欠損値を処理する」→ 具体的にどう？
df.fillna(???)

# 「特徴量を作る」→ 何を作る？どのルール？
df['new_feature'] = ???

# 「良いモデルを作る」→ 何を持って「良い」？
model = ???
```

→ **形式手法で制約を明確化**

---

## 2. KAOS図によるゴール構造化

### ルートゴール

```
ROOT: Kaggleコンペティションで高スコアを達成する
```

### KAOS階層構造

```
ROOT: 高スコア達成
├─ G1: データ品質を保証する【AND】
│   ├─ G11: 欠損値を適切に処理
│   │   ├─ G111: 数値カラムの欠損値処理
│   │   └─ G112: カテゴリカルカラムの欠損値処理
│   ├─ G12: 外れ値を検出・処理
│   └─ G13: データ型の整合性を保証
│
├─ G2: 有効な特徴量を生成する【AND】
│   ├─ G21: ドメイン知識に基づく特徴量
│   ├─ G22: 統計的特徴量
│   └─ G23: 相互作用特徴量
│
├─ G3: 適切なモデルを構築する【AND】
│   ├─ G31: ベースラインモデル作成
│   ├─ G32: ハイパーパラメータ最適化
│   └─ G33: アンサンブル構築
│
└─ G4: 評価とイテレーションを実施【AND】
    ├─ G41: クロスバリデーション実施
    ├─ G42: リーダーボードとの乖離確認
    └─ G43: 改善サイクル実行
```

### NetworkXによる可視化コード

```python
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']

# グラフ作成
G = nx.DiGraph()

# ノード追加
nodes = {
    'ROOT': {'label': 'ROOT: 高スコア達成', 'color': 'lightpink'},
    
    # G1系統
    'G1': {'label': 'G1: データ品質保証', 'color': 'lightblue'},
    'G11': {'label': 'G11: 欠損値処理', 'color': 'lightblue'},
    'G111': {'label': 'G111: 数値欠損値', 'color': 'lightblue'},
    'G112': {'label': 'G112: カテゴリ欠損値', 'color': 'lightblue'},
    'G12': {'label': 'G12: 外れ値処理', 'color': 'lightblue'},
    'G13': {'label': 'G13: 型整合性', 'color': 'lightblue'},
    
    # G2系統
    'G2': {'label': 'G2: 特徴量生成', 'color': 'lightgreen'},
    'G21': {'label': 'G21: ドメイン特徴量', 'color': 'lightgreen'},
    'G22': {'label': 'G22: 統計特徴量', 'color': 'lightgreen'},
    'G23': {'label': 'G23: 相互作用特徴量', 'color': 'lightgreen'},
    
    # G3系統
    'G3': {'label': 'G3: モデル構築', 'color': 'lightyellow'},
    'G31': {'label': 'G31: ベースライン', 'color': 'lightyellow'},
    'G32': {'label': 'G32: 最適化', 'color': 'lightyellow'},
    'G33': {'label': 'G33: アンサンブル', 'color': 'lightyellow'},
    
    # G4系統
    'G4': {'label': 'G4: 評価・改善', 'color': 'lavender'},
    'G41': {'label': 'G41: CV実施', 'color': 'lavender'},
    'G42': {'label': 'G42: LB確認', 'color': 'lavender'},
    'G43': {'label': 'G43: 改善サイクル', 'color': 'lavender'},
}

for node, attrs in nodes.items():
    G.add_node(node, **attrs)

# エッジ追加
edges = [
    ('ROOT', 'G1'), ('ROOT', 'G2'), ('ROOT', 'G3'), ('ROOT', 'G4'),
    ('G1', 'G11'), ('G1', 'G12'), ('G1', 'G13'),
    ('G11', 'G111'), ('G11', 'G112'),
    ('G2', 'G21'), ('G2', 'G22'), ('G2', 'G23'),
    ('G3', 'G31'), ('G3', 'G32'), ('G3', 'G33'),
    ('G4', 'G41'), ('G4', 'G42'), ('G4', 'G43'),
]

G.add_edges_from(edges)

# 描画
pos = nx.spring_layout(G, k=2, iterations=50)
plt.figure(figsize=(16, 12))

colors = [G.nodes[node]['color'] for node in G.nodes()]
labels = {node: G.nodes[node]['label'] for node in G.nodes()}

nx.draw(G, pos, 
        node_color=colors,
        labels=labels,
        node_size=3000,
        font_size=9,
        font_weight='bold',
        arrows=True,
        arrowsize=20,
        edge_color='gray',
        linewidths=2,
        width=2)

plt.title("KAOS: Kaggle Competition Goal Structure", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('kaggle_kaos.png', dpi=300, bbox_inches='tight')
plt.show()

print("KAOS図を生成しました: kaggle_kaos.png")
```

---

## 3. Alloy形式記法による制約記述

### Alloyモデル全体

```alloy
/**
 * Kaggleコンペティション用AI-Augmented形式手法
 * タイタニック生存予測の例
 */

module KaggleCompetition

// ============================================
// 基本シグネチャ定義
// ============================================

/**
 * データセット
 */
sig Dataset {
  train: one TrainData,
  test: one TestData
}

/**
 * 訓練データ
 */
sig TrainData {
  rows: set Row,
  targetColumn: one TargetColumn
}

/**
 * テストデータ
 */
sig TestData {
  rows: set Row
}

/**
 * データ行
 */
sig Row {
  features: set Feature,
  missingValues: set Feature  // 欠損値を持つ特徴量
}

/**
 * 特徴量
 */
abstract sig Feature {}

sig NumericalFeature extends Feature {
  value: lone Int  // 欠損の可能性があるのでlone
}

sig CategoricalFeature extends Feature {
  category: lone String  // 欠損の可能性があるのでlone
}

/**
 * ターゲット変数（生存: 0 or 1）
 */
sig TargetColumn {
  value: one Int
}

/**
 * 前処理済みデータ
 */
sig ProcessedData {
  originalData: one Dataset,
  processedRows: set Row
}

/**
 * 特徴量エンジニアリング後のデータ
 */
sig FeatureEngineeredData {
  baseData: one ProcessedData,
  newFeatures: set Feature
}

/**
 * モデル
 */
abstract sig Model {}

sig LogisticRegression extends Model {}
sig RandomForest extends Model {}
sig GradientBoosting extends Model {}

/**
 * 訓練済みモデル
 */
sig TrainedModel {
  model: one Model,
  trainingData: one FeatureEngineeredData,
  hyperparameters: one HyperParameters
}

/**
 * ハイパーパラメータ
 */
sig HyperParameters {
  learningRate: lone Float,
  maxDepth: lone Int,
  nEstimators: lone Int
}

/**
 * 予測結果
 */
sig Prediction {
  model: one TrainedModel,
  testData: one TestData,
  predictions: seq Int  // 0 or 1の系列
}

/**
 * 評価指標
 */
sig Evaluation {
  accuracy: one Float,
  precision: one Float,
  recall: one Float,
  f1Score: one Float
}

// ============================================
// 制約条件（Facts）
// ============================================

/**
 * G11: 欠損値処理の制約
 */
fact MissingValueHandling {
  // 前処理後のデータには欠損値がない
  all pd: ProcessedData |
    no pd.processedRows.missingValues
}

/**
 * G111: 数値特徴量の欠損値処理
 */
fact NumericalMissingValues {
  all pd: ProcessedData, r: pd.processedRows, f: NumericalFeature |
    f in r.features implies some f.value
}

/**
 * G112: カテゴリカル特徴量の欠損値処理
 */
fact CategoricalMissingValues {
  all pd: ProcessedData, r: pd.processedRows, f: CategoricalFeature |
    f in r.features implies some f.category
}

/**
 * G13: データ型の整合性
 */
fact DataTypeConsistency {
  // 同じ特徴量は同じ型
  all r1, r2: Row, f1, f2: Feature |
    (f1 in r1.features and f2 in r2.features and f1 = f2) implies
      (f1 in NumericalFeature iff f2 in NumericalFeature)
}

/**
 * G12: 外れ値の制約（例：年齢は0-120）
 */
fact OutlierConstraints {
  all f: NumericalFeature |
    // 年齢の特徴量の場合
    some f.value implies
      (f.value >= 0 and f.value <= 120)
}

/**
 * ターゲット変数は0または1
 */
fact TargetBinary {
  all t: TargetColumn |
    t.value = 0 or t.value = 1
}

/**
 * G21-G23: 特徴量エンジニアリングの制約
 */
fact FeatureEngineeringRules {
  all fed: FeatureEngineeredData |
    // 新しい特徴量は既存データから生成される
    fed.newFeatures in fed.baseData.processedRows.features or
    // または既存特徴量の組み合わせ
    some f1, f2: fed.baseData.processedRows.features |
      fed.newFeatures in (f1 + f2)
}

/**
 * G32: ハイパーパラメータの妥当な範囲
 */
fact HyperparameterConstraints {
  all hp: HyperParameters |
    // 学習率は0より大きく1より小さい
    (some hp.learningRate implies 
      (hp.learningRate > 0.0 and hp.learningRate < 1.0)) and
    // 木の深さは1以上20以下
    (some hp.maxDepth implies
      (hp.maxDepth >= 1 and hp.maxDepth <= 20)) and
    // 推定器の数は1以上1000以下
    (some hp.nEstimators implies
      (hp.nEstimators >= 1 and hp.nEstimators <= 1000))
}

/**
 * G31: 訓練データとテストデータの分離
 */
fact TrainTestSeparation {
  all ds: Dataset |
    no ds.train.rows & ds.test.rows
}

/**
 * 予測数とテストデータの行数が一致
 */
fact PredictionCountMatches {
  all p: Prediction |
    #p.predictions = #p.testData.rows
}

/**
 * 予測値は0または1
 */
fact PredictionBinary {
  all p: Prediction, i: Int |
    i in p.predictions.inds implies
      (p.predictions[i] = 0 or p.predictions[i] = 1)
}

/**
 * G41: 評価指標の妥当な範囲（0〜1）
 */
fact EvaluationMetricsRange {
  all e: Evaluation |
    e.accuracy >= 0.0 and e.accuracy <= 1.0 and
    e.precision >= 0.0 and e.precision <= 1.0 and
    e.recall >= 0.0 and e.recall <= 1.0 and
    e.f1Score >= 0.0 and e.f1Score <= 1.0
}

// ============================================
// 述語（Predicates）
// ============================================

/**
 * G1達成：データ品質が保証されている
 */
pred G1_Achieved {
  // すべてのProcessedDataで欠損値がない
  all pd: ProcessedData |
    no pd.processedRows.missingValues and
    // すべての数値特徴量に値がある
    (all r: pd.processedRows, f: NumericalFeature |
      f in r.features implies some f.value) and
    // すべてのカテゴリカル特徴量にカテゴリがある
    (all r: pd.processedRows, f: CategoricalFeature |
      f in r.features implies some f.category)
}

/**
 * G2達成：特徴量エンジニアリング完了
 */
pred G2_Achieved {
  some fed: FeatureEngineeredData |
    // 新しい特徴量が生成されている
    some fed.newFeatures and
    // ベースデータは前処理済み
    G1_Achieved
}

/**
 * G3達成：モデル構築完了
 */
pred G3_Achieved {
  some tm: TrainedModel |
    // モデルが訓練されている
    some tm.model and
    // 特徴量エンジニアリング済みデータを使用
    G2_Achieved and
    // ハイパーパラメータが妥当な範囲
    validHyperparameters[tm.hyperparameters]
}

/**
 * ハイパーパラメータの妥当性チェック
 */
pred validHyperparameters[hp: HyperParameters] {
  (some hp.learningRate implies 
    hp.learningRate > 0.0 and hp.learningRate < 1.0) and
  (some hp.maxDepth implies
    hp.maxDepth >= 1 and hp.maxDepth <= 20) and
  (some hp.nEstimators implies
    hp.nEstimators >= 1 and hp.nEstimators <= 1000)
}

/**
 * G4達成：評価・改善サイクル実施
 */
pred G4_Achieved {
  some e: Evaluation |
    // 評価指標が算出されている
    e.accuracy > 0.0 and
    // モデル構築が完了している
    G3_Achieved
}

/**
 * 完全なKaggleパイプライン
 */
pred CompleteKagglePipeline {
  G1_Achieved and
  G2_Achieved and
  G3_Achieved and
  G4_Achieved
}

/**
 * 高スコア達成の条件
 */
pred HighScoreAchieved {
  CompleteKagglePipeline and
  some e: Evaluation |
    e.accuracy > 0.8 and  // 80%以上の精度
    e.f1Score > 0.75      // F1スコア75%以上
}

// ============================================
// アサーション（Assertions）
// ============================================

/**
 * 前処理後には欠損値がない
 */
assert NoMissingAfterProcessing {
  all pd: ProcessedData |
    no pd.processedRows.missingValues
}

/**
 * 予測値は常にバイナリ
 */
assert PredictionsAreBinary {
  all p: Prediction, i: Int |
    i in p.predictions.inds implies
      (p.predictions[i] = 0 or p.predictions[i] = 1)
}

/**
 * 訓練データとテストデータは重複しない
 */
assert NoTrainTestOverlap {
  all ds: Dataset |
    no ds.train.rows & ds.test.rows
}

/**
 * ハイパーパラメータは常に妥当な範囲
 */
assert ValidHyperparameters {
  all hp: HyperParameters |
    (some hp.learningRate implies 
      hp.learningRate > 0.0 and hp.learningRate < 1.0) and
    (some hp.maxDepth implies
      hp.maxDepth >= 1 and hp.maxDepth <= 20)
}

/**
 * 高スコア達成には全ゴールが必要
 */
assert HighScoreRequiresAllGoals {
  HighScoreAchieved implies
    (G1_Achieved and G2_Achieved and G3_Achieved and G4_Achieved)
}

// ============================================
// コマンド（Commands）
// ============================================

/**
 * G1達成シナリオ
 */
run G1_Achieved for 5

/**
 * 完全なパイプライン実行
 */
run CompleteKagglePipeline for 5

/**
 * 高スコア達成シナリオ
 */
run HighScoreAchieved for 5

/**
 * アサーション検証
 */
check NoMissingAfterProcessing for 10
check PredictionsAreBinary for 10
check NoTrainTestOverlap for 10
check ValidHyperparameters for 10
check HighScoreRequiresAllGoals for 10
```

---

## 4. Claude Codeへの実装プロンプト

### プロンプト生成（AI-Augmented）

**人間からClaude Desktopへの指示**：

```
このAlloyモデル（kaggle_competition.als）を読んで、
Kaggleタイタニック生存予測のPython実装プロンプトを生成してください。

含めるもの：
1. 各Goalの実装要件
2. Alloyで定義された制約のバリデーション
3. テスト観点
4. コード構造

対象：Claude Code実装用
```

**AIが生成するClaude Codeプロンプト**：

```markdown
# Kaggleタイタニック生存予測 - 実装仕様

## プロジェクト概要
Alloyモデルで形式化された制約に基づき、
タイタニック生存予測パイプラインを構築します。

## ディレクトリ構造
```
kaggle-titanic/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── preprocessing.py    # G1: データ品質保証
│   ├── features.py         # G2: 特徴量生成
│   ├── models.py           # G3: モデル構築
│   ├── evaluation.py       # G4: 評価
│   └── validation.py       # Alloy制約検証
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_validation.py
└── main.py
```

## G1: データ品質保証の実装

### 制約（Alloy fact より）
```python
# fact MissingValueHandling
# fact NumericalMissingValues
# fact CategoricalMissingValues
# fact OutlierConstraints
```

### 実装要件

**preprocessing.py**:
```python
class DataPreprocessor:
    def __init__(self):
        # 欠損値処理ルール
        self.numerical_strategy = 'median'
        self.categorical_strategy = 'mode'
    
    def handle_missing_values(self, df):
        """
        制約: 処理後のデータに欠損値がない
        Alloy: fact MissingValueHandling
        """
        # G111: 数値特徴量の欠損値処理
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # G112: カテゴリカル特徴量の欠損値処理
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # 検証: 欠損値が残っていないこと
        assert df.isnull().sum().sum() == 0, "欠損値が残っています"
        
        return df
    
    def handle_outliers(self, df, column, min_val, max_val):
        """
        制約: 外れ値を妥当な範囲に収める
        Alloy: fact OutlierConstraints
        例: 年齢は0-120
        """
        df[column] = df[column].clip(min_val, max_val)
        
        # 検証
        assert df[column].min() >= min_val, f"{column}の最小値が範囲外"
        assert df[column].max() <= max_val, f"{column}の最大値が範囲外"
        
        return df
```

### テスト観点
```python
# tests/test_preprocessing.py
def test_no_missing_after_processing():
    """Alloy assertion: NoMissingAfterProcessing"""
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(sample_data)
    assert df.isnull().sum().sum() == 0

def test_outlier_constraints():
    """Alloy fact: OutlierConstraints"""
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_outliers(sample_data, 'Age', 0, 120)
    assert df['Age'].min() >= 0
    assert df['Age'].max() <= 120
```

## G2: 特徴量生成の実装

### 制約（Alloy fact より）
```python
# fact FeatureEngineeringRules
```

### 実装要件

**features.py**:
```python
class FeatureEngineer:
    def create_features(self, df):
        """
        制約: 新特徴量は既存特徴量から生成
        Alloy: fact FeatureEngineeringRules
        """
        # G21: ドメイン知識に基づく特徴量
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # G22: 統計的特徴量
        df['Age_binned'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 120], 
                                  labels=['child', 'teen', 'adult', 'senior'])
        
        # G23: 相互作用特徴量
        df['Fare_per_person'] = df['Fare'] / df['FamilySize']
        
        # 検証: 新特徴量が生成されていること
        required_features = ['FamilySize', 'IsAlone', 'Age_binned', 'Fare_per_person']
        for feature in required_features:
            assert feature in df.columns, f"特徴量 {feature} が生成されていません"
        
        return df
```

### テスト観点
```python
def test_feature_generation():
    """Alloy pred: G2_Achieved"""
    fe = FeatureEngineer()
    df = fe.create_features(sample_data)
    
    # 新特徴量が存在する
    assert 'FamilySize' in df.columns
    assert 'IsAlone' in df.columns
    
    # 特徴量の値が妥当
    assert df['FamilySize'].min() >= 1
    assert df['IsAlone'].isin([0, 1]).all()
```

## G3: モデル構築の実装

### 制約（Alloy fact より）
```python
# fact HyperparameterConstraints
# fact TrainTestSeparation
```

### 実装要件

**models.py**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.hyperparameters = {}
    
    def set_hyperparameters(self, **kwargs):
        """
        制約: ハイパーパラメータが妥当な範囲内
        Alloy: fact HyperparameterConstraints
        """
        if 'learning_rate' in kwargs:
            lr = kwargs['learning_rate']
            assert 0.0 < lr < 1.0, f"学習率は0-1の範囲: {lr}"
            self.hyperparameters['learning_rate'] = lr
        
        if 'max_depth' in kwargs:
            depth = kwargs['max_depth']
            assert 1 <= depth <= 20, f"max_depthは1-20の範囲: {depth}"
            self.hyperparameters['max_depth'] = depth
        
        if 'n_estimators' in kwargs:
            n_est = kwargs['n_estimators']
            assert 1 <= n_est <= 1000, f"n_estimatorsは1-1000の範囲: {n_est}"
            self.hyperparameters['n_estimators'] = n_est
    
    def train(self, X, y):
        """
        制約: 訓練データとテストデータの分離
        Alloy: fact TrainTestSeparation
        """
        # G31: ベースラインモデル
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.hyperparameters)
        
        # 訓練
        self.model.fit(X, y)
        
        return self.model
    
    def predict(self, X):
        """
        制約: 予測値は0または1
        Alloy: fact PredictionBinary
        """
        predictions = self.model.predict(X)
        
        # 検証
        assert set(predictions).issubset({0, 1}), "予測値は0または1である必要があります"
        
        return predictions
```

### テスト観点
```python
def test_hyperparameter_validation():
    """Alloy assertion: ValidHyperparameters"""
    trainer = ModelTrainer()
    
    # 正常系
    trainer.set_hyperparameters(learning_rate=0.1, max_depth=10)
    
    # 異常系：範囲外の値
    with pytest.raises(AssertionError):
        trainer.set_hyperparameters(learning_rate=1.5)  # > 1.0
    
    with pytest.raises(AssertionError):
        trainer.set_hyperparameters(max_depth=25)  # > 20

def test_prediction_binary():
    """Alloy assertion: PredictionsAreBinary"""
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    predictions = trainer.predict(X_test)
    
    assert set(predictions).issubset({0, 1})
```

## G4: 評価・改善の実装

### 制約（Alloy fact より）
```python
# fact EvaluationMetricsRange
```

### 実装要件

**evaluation.py**:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

class ModelEvaluator:
    def evaluate(self, y_true, y_pred):
        """
        制約: 評価指標は0-1の範囲
        Alloy: fact EvaluationMetricsRange
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        # 検証: すべての指標が0-1の範囲内
        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{metric_name}が範囲外: {value}"
        
        return metrics
    
    def cross_validate(self, model, X, y, cv=5):
        """
        G41: クロスバリデーション実施
        """
        cv_scores = cross_val_score(model, X, y, cv=cv)
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
```

### テスト観点
```python
def test_evaluation_metrics_range():
    """Alloy fact: EvaluationMetricsRange"""
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    
    for metric_name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{metric_name}が範囲外"
```

## 統合検証

**validation.py**:
```python
class AlloyConstraintValidator:
    """Alloy制約の統合検証"""
    
    @staticmethod
    def validate_pipeline(train_df, test_df, predictions):
        """完全なパイプラインの検証"""
        
        # G1: データ品質
        assert train_df.isnull().sum().sum() == 0, "訓練データに欠損値"
        assert test_df.isnull().sum().sum() == 0, "テストデータに欠損値"
        
        # G3: 訓練・テストの分離
        # （実際にはIndexで確認）
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        assert len(train_indices & test_indices) == 0, "訓練とテストが重複"
        
        # 予測値の検証
        assert len(predictions) == len(test_df), "予測数が不一致"
        assert set(predictions).issubset({0, 1}), "予測値は0または1"
        
        print("✅ すべてのAlloy制約を満たしています")
        return True
```

## main.py: 全体フロー

```python
def main():
    # データ読み込み
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    # G1: データ前処理
    preprocessor = DataPreprocessor()
    train_df = preprocessor.handle_missing_values(train_df)
    train_df = preprocessor.handle_outliers(train_df, 'Age', 0, 120)
    
    test_df = preprocessor.handle_missing_values(test_df)
    test_df = preprocessor.handle_outliers(test_df, 'Age', 0, 120)
    
    # G2: 特徴量エンジニアリング
    fe = FeatureEngineer()
    train_df = fe.create_features(train_df)
    test_df = fe.create_features(test_df)
    
    # G3: モデル訓練
    X_train = train_df.drop(['Survived'], axis=1)
    y_train = train_df['Survived']
    
    trainer = ModelTrainer('random_forest')
    trainer.set_hyperparameters(max_depth=10, n_estimators=100)
    model = trainer.train(X_train, y_train)
    
    # G4: 評価
    predictions = trainer.predict(X_train)
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_train, predictions)
    
    print("評価結果:", metrics)
    
    # 予測
    X_test = test_df
    test_predictions = trainer.predict(X_test)
    
    # 統合検証
    validator = AlloyConstraintValidator()
    validator.validate_pipeline(train_df, test_df, test_predictions)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    
    print("✅ パイプライン完了")

if __name__ == '__main__':
    main()
```

## テスト実行

```bash
# すべてのテスト実行
pytest tests/

# Alloy制約の検証
pytest tests/test_validation.py -v

# カバレッジ確認
pytest --cov=src tests/
```

## 期待される効果

1. **制約違反の早期発見**
   - 欠損値が残っている → AssertionError
   - ハイパーパラメータが範囲外 → AssertionError
   
2. **実装の一貫性**
   - Alloyモデルと実装が1対1対応
   - テストがAlloy制約を直接検証

3. **ドキュメント自動生成**
   - Alloyモデルが仕様書
   - コードコメントにAlloy参照

4. **AIへの指示精度向上**
   - 曖昧性がない明確な仕様
   - Claude Codeが正確に実装
```

---

## 5. 実装例とテスト

上記のプロンプトをClaude Codeに渡すことで、Alloy制約を満たす実装が自動生成されます。

### 実行例

```bash
$ claude

> 上記の実装仕様に従って、Kaggleタイタニック予測パイプラインを実装してください。
  各ファイルを順番に作成し、テストも含めてください。
```

---

## 6. 効果の検証

### 従来のアプローチとの比較

| 観点 | 従来（自然言語のみ） | AI-Augmented形式手法 |
|------|-------------------|---------------------|
| **曖昧性** | 「欠損値を処理する」→ 方法不明 | Alloy factで明確に定義 |
| **制約漏れ** | ハイパーパラメータの範囲チェック漏れ | Alloy assertionで自動検証 |
| **テスト観点** | 人間が考える（漏れあり） | Alloy制約から自動生成 |
| **実装の一貫性** | ファイル間で不整合の可能性 | Alloyモデルが唯一の真実 |
| **AIへの指示** | 解釈のブレ | 形式的に検証済み |

### 定量的効果（推定）

```
バグ発見時期の前倒し：
  従来: 実装後のテストで発見
  形式手法: Alloy検証時に発見（開発初期）
  
工数削減：
  要件定義の手戻り: 50%削減
  実装の手戻り: 30%削減
  
品質向上：
  制約違反の検出率: 95%以上
```

---

## まとめ

### AI-Augmented形式手法のKaggleへの適用価値

1. **データ品質の保証**
   - 欠損値、外れ値の処理ルールを形式化
   - 処理後のデータが制約を満たすことを保証

2. **特徴量エンジニアリングの体系化**
   - 特徴量生成ルールを明示
   - 依存関係を可視化

3. **モデル構築の標準化**
   - ハイパーパラメータの妥当な範囲を定義
   - 訓練・テストの分離を保証

4. **評価の厳密化**
   - 評価指標の妥当性を検証
   - クロスバリデーションの実施を保証

### 次のステップ

1. 実際のKaggleコンペで実践
2. ブログ記事化（Zenn投稿）
3. コミュニティフィードバック
4. より複雑なコンペへの拡張

---

**作成日**: 2026年1月26日  
**対象**: 古閑弘晃さんのKaggle学習
