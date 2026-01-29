/*/**
 * Kaggleコンペティション用AI-Augmented形式手法（修正版）
 * タイタニック生存予測の例
 * 
 * 修正点：
 * - Float型を削除（Alloyは浮動小数点をサポートしない）
 * - 評価指標を整数パーセンテージで表現（0-100）
 * - 学習率などは抽象的に表現
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
 * ハイパーパラメータ
 * 注：学習率は抽象的に表現（Small/Medium/Large）
 */
abstract sig LearningRateLevel {}
one sig SmallLR extends LearningRateLevel {}
one sig MediumLR extends LearningRateLevel {}
one sig LargeLR extends LearningRateLevel {}

sig HyperParameters {
  learningRateLevel: lone LearningRateLevel,
  maxDepth: lone Int,
  nEstimators: lone Int
}

/**
 * 訓練済みモデル
 */
sig TrainedModel {
  model: one Model,
  trainingData: one FeatureEngineeredData,
  hyperparameters: one HyperParameters
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
 * 評価指標（パーセンテージ: 0-100）
 */
sig Evaluation {
  accuracyPercent: one Int,    // 0-100
  precisionPercent: one Int,   // 0-100
  recallPercent: one Int,      // 0-100
  f1ScorePercent: one Int      // 0-100
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
    // 年齢の特徴量の場合（値が存在する場合）
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
 * G41: 評価指標の妥当な範囲（0〜100パーセント）
 */
fact EvaluationMetricsRange {
  all e: Evaluation |
    e.accuracyPercent >= 0 and e.accuracyPercent <= 100 and
    e.precisionPercent >= 0 and e.precisionPercent <= 100 and
    e.recallPercent >= 0 and e.recallPercent <= 100 and
    e.f1ScorePercent >= 0 and e.f1ScorePercent <= 100
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
    e.accuracyPercent > 0 and
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
 * 高スコア達成の条件（80%以上の精度）
 */
pred HighScoreAchieved {
  CompleteKagglePipeline and
  some e: Evaluation |
    e.accuracyPercent >= 80 and  // 80%以上の精度
    e.f1ScorePercent >= 75       // F1スコア75%以上
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
    (some hp.maxDepth implies
      hp.maxDepth >= 1 and hp.maxDepth <= 20) and
    (some hp.nEstimators implies
      hp.nEstimators >= 1 and hp.nEstimators <= 1000)
}

/**
 * 高スコア達成には全ゴールが必要
 */
assert HighScoreRequiresAllGoals {
  HighScoreAchieved implies
    (G1_Achieved and G2_Achieved and G3_Achieved and G4_Achieved)
}

/**
 * 評価指標は常に0-100の範囲
 */
assert EvaluationInRange {
  all e: Evaluation |
    e.accuracyPercent >= 0 and e.accuracyPercent <= 100 and
    e.precisionPercent >= 0 and e.precisionPercent <= 100
}

// ============================================
// コマンド（Commands）
// ============================================

/**
 * G1達成シナリオ
 */
run G1_Achieved for 3

/**
 * 完全なパイプライン実行
 */
run CompleteKagglePipeline for 3

/**
 * 高スコア達成シナリオ
 */
run HighScoreAchieved for 3

/**
 * アサーション検証
 */
check NoMissingAfterProcessing for 5
check PredictionsAreBinary for 5
check NoTrainTestOverlap for 5
check ValidHyperparameters for 5
check HighScoreRequiresAllGoals for 5
check EvaluationInRange for 5

/**
 * 最小限のパイプライン（デバッグ用）
 */
run {
  some pd: ProcessedData |
    no pd.processedRows.missingValues
} for 2
/**
 * G2達成シナリオ
 */
run G2_Achieved for 3

/**
 * G3達成シナリオ
 */
run G3_Achieved for 3

/**
 * G4達成シナリオ
 */
run G4_Achieved for 3
