/**
 * Kaggleコンペティション用AI-Augmented形式手法（修正版v2）
 * タイタニック生存予測の例
 * 
 * 修正履歴：
 * v1: Float型削除、整数パーセンテージ化
 * v2: FeatureEngineeringRules を簡素化（過剰制約を解消）
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
  predictions: seq Int
}

/**
 * 評価指標（パーセンテージ: 0-100）
 */
sig Evaluation {
  accuracyPercent: one Int,
  precisionPercent: one Int,
  recallPercent: one Int,
  f1ScorePercent: one Int
}

// ============================================
// 制約条件（Facts）
// ============================================

/**
 * G11: 欠損値処理の制約
 */
fact MissingValueHandling {
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
  all r1, r2: Row, f1, f2: Feature |
    (f1 in r1.features and f2 in r2.features and f1 = f2) implies
      (f1 in NumericalFeature iff f2 in NumericalFeature)
}

/**
 * G12: 外れ値の制約（例：年齢は0-120）
 */
fact OutlierConstraints {
  all f: NumericalFeature |
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
 * G21-G23: 特徴量エンジニアリングの制約（修正版）
 * 
 * 修正内容：
 * - 過剰に詳細な制約を削除
 * - 抽象的なレベルで表現
 * - 「新しい特徴量が存在する」ことのみを要求
 */
fact FeatureEngineeringRules {
  all fed: FeatureEngineeredData |
    // 特徴量エンジニアリングが実施されている
    // （具体的な生成方法は実装レベルで決定）
    some fed.newFeatures
}

/**
 * G32: ハイパーパラメータの妥当な範囲
 */
fact HyperparameterConstraints {
  all hp: HyperParameters |
    (some hp.maxDepth implies
      (hp.maxDepth >= 1 and hp.maxDepth <= 20)) and
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
  all pd: ProcessedData |
    no pd.processedRows.missingValues and
    (all r: pd.processedRows, f: NumericalFeature |
      f in r.features implies some f.value) and
    (all r: pd.processedRows, f: CategoricalFeature |
      f in r.features implies some f.category)
}

/**
 * G2達成：特徴量エンジニアリング完了
 */
pred G2_Achieved {
  some fed: FeatureEngineeredData |
    some fed.newFeatures and
    G1_Achieved
}

/**
 * G3達成：モデル構築完了
 */
pred G3_Achieved {
  some tm: TrainedModel |
    some tm.model and
    G2_Achieved and
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
    e.accuracyPercent > 0 and
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
    e.accuracyPercent >= 80 and
    e.f1ScorePercent >= 75
}

// ============================================
// アサーション（Assertions）
// ============================================

assert NoMissingAfterProcessing {
  all pd: ProcessedData |
    no pd.processedRows.missingValues
}

assert PredictionsAreBinary {
  all p: Prediction, i: Int |
    i in p.predictions.inds implies
      (p.predictions[i] = 0 or p.predictions[i] = 1)
}

assert NoTrainTestOverlap {
  all ds: Dataset |
    no ds.train.rows & ds.test.rows
}

assert ValidHyperparameters {
  all hp: HyperParameters |
    (some hp.maxDepth implies
      hp.maxDepth >= 1 and hp.maxDepth <= 20) and
    (some hp.nEstimators implies
      hp.nEstimators >= 1 and hp.nEstimators <= 1000)
}

assert HighScoreRequiresAllGoals {
  HighScoreAchieved implies
    (G1_Achieved and G2_Achieved and G3_Achieved and G4_Achieved)
}

assert EvaluationInRange {
  all e: Evaluation |
    e.accuracyPercent >= 0 and e.accuracyPercent <= 100 and
    e.precisionPercent >= 0 and e.precisionPercent <= 100
}

// ============================================
// コマンド（Commands）
// ============================================

/**
 * 個別ゴールのテスト
 */
run G1_Achieved for 3
run G2_Achieved for 3
run G3_Achieved for 3
run G4_Achieved for 3

/**
 * 段階的な組み合わせ
 */
run { G1_Achieved and G2_Achieved } for 3
run { G1_Achieved and G2_Achieved and G3_Achieved } for 3

/**
 * 完全なパイプライン
 */
run CompleteKagglePipeline for 3
run HighScoreAchieved for 3

/**
 * より大きなスコープでの検証
 */
run CompleteKagglePipeline for 5
run HighScoreAchieved for 5

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
 * デバッグ用最小パイプライン
 */
run {
  some pd: ProcessedData |
    no pd.processedRows.missingValues
} for 2
