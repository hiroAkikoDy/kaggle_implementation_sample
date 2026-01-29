"""
Kaggle AI-Augmented形式手法 - 実装サンプル
Alloy制約に基づくデータ前処理・特徴量生成・モデル訓練

対象: タイタニック生存予測
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AlloyConstraintValidator:
    """
    Alloy形式記法で定義された制約をPythonで検証
    
    対応するAlloy facts:
    - MissingValueHandling
    - OutlierConstraints
    - HyperparameterConstraints
    - PredictionBinary
    - EvaluationMetricsRange
    """
    
    @staticmethod
    def validate_no_missing(df: pd.DataFrame, stage: str = "") -> bool:
        """
        Alloy: fact MissingValueHandling
        制約: 前処理後のデータには欠損値がない
        """
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            raise ValueError(f"{stage}: 欠損値が{missing_count}個残っています")
        print(f"✅ {stage}: 欠損値チェック合格")
        return True
    
    @staticmethod
    def validate_outliers(df: pd.DataFrame, column: str, min_val: float, max_val: float) -> bool:
        """
        Alloy: fact OutlierConstraints
        制約: 値が妥当な範囲内
        """
        actual_min = df[column].min()
        actual_max = df[column].max()
        
        if actual_min < min_val or actual_max > max_val:
            raise ValueError(
                f"{column}の範囲制約違反: 期待[{min_val}, {max_val}], "
                f"実際[{actual_min}, {actual_max}]"
            )
        print(f"✅ {column}: 外れ値制約合格 [{actual_min:.2f}, {actual_max:.2f}]")
        return True
    
    @staticmethod
    def validate_hyperparameters(params: Dict) -> bool:
        """
        Alloy: fact HyperparameterConstraints
        制約: ハイパーパラメータが妥当な範囲
        """
        if 'learning_rate' in params:
            lr = params['learning_rate']
            if not (0.0 < lr < 1.0):
                raise ValueError(f"learning_rateは(0, 1)の範囲: {lr}")
        
        if 'max_depth' in params:
            depth = params['max_depth']
            if not (1 <= depth <= 20):
                raise ValueError(f"max_depthは[1, 20]の範囲: {depth}")
        
        if 'n_estimators' in params:
            n_est = params['n_estimators']
            if not (1 <= n_est <= 1000):
                raise ValueError(f"n_estimatorsは[1, 1000]の範囲: {n_est}")
        
        print(f"✅ ハイパーパラメータ検証合格: {params}")
        return True
    
    @staticmethod
    def validate_predictions_binary(predictions: np.ndarray) -> bool:
        """
        Alloy: fact PredictionBinary
        制約: 予測値は0または1
        """
        unique_values = set(predictions)
        if not unique_values.issubset({0, 1}):
            raise ValueError(f"予測値は0または1である必要があります: {unique_values}")
        print(f"✅ 予測値バイナリチェック合格")
        return True
    
    @staticmethod
    def validate_metrics_range(metrics: Dict[str, float]) -> bool:
        """
        Alloy: fact EvaluationMetricsRange
        制約: 評価指標は[0, 1]の範囲
        """
        for metric_name, value in metrics.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{metric_name}が範囲外[0, 1]: {value}")
        print(f"✅ 評価指標範囲チェック合格")
        return True
    
    @staticmethod
    def validate_train_test_separation(train_indices: set, test_indices: set) -> bool:
        """
        Alloy: fact TrainTestSeparation
        制約: 訓練データとテストデータは重複しない
        """
        overlap = train_indices & test_indices
        if len(overlap) > 0:
            raise ValueError(f"訓練とテストが{len(overlap)}行重複しています")
        print(f"✅ 訓練・テスト分離チェック合格")
        return True


class DataPreprocessor:
    """
    G1: データ品質保証
    
    対応するKAOS Goal:
    - G11: 欠損値処理
    - G12: 外れ値処理
    - G13: データ型整合性
    """
    
    def __init__(self, validator: AlloyConstraintValidator):
        self.validator = validator
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        G111, G112: 欠損値処理
        
        Alloy constraints:
        - fact NumericalMissingValues
        - fact CategoricalMissingValues
        """
        print("\n[G11] 欠損値処理開始...")
        
        # 数値特徴量（中央値で補完）
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  - {col}: 中央値{median_val:.2f}で補完")
        
        # カテゴリカル特徴量（最頻値で補完）
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"  - {col}: 最頻値'{mode_val}'で補完")
        
        # Alloy制約検証
        self.validator.validate_no_missing(df, "欠損値処理後")
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        G12: 外れ値処理
        
        Alloy constraint: fact OutlierConstraints
        """
        print("\n[G12] 外れ値処理開始...")
        
        # 年齢: 0-120歳の範囲に制限
        if 'Age' in df.columns:
            df['Age'] = df['Age'].clip(0, 120)
            self.validator.validate_outliers(df, 'Age', 0, 120)
        
        # 運賃: 0以上
        if 'Fare' in df.columns:
            df['Fare'] = df['Fare'].clip(0, None)
            print(f"✅ Fare: 負の値を0にクリップ")
        
        return df


class FeatureEngineer:
    """
    G2: 特徴量生成
    
    対応するKAOS Goal:
    - G21: ドメイン知識特徴量
    - G22: 統計的特徴量
    - G23: 相互作用特徴量
    """
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alloy constraint: fact FeatureEngineeringRules
        新特徴量は既存特徴量から生成される
        """
        print("\n[G2] 特徴量エンジニアリング開始...")
        
        # G21: ドメイン知識に基づく特徴量
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            print(f"  - FamilySize, IsAlone生成（ドメイン知識）")
        
        # G22: 統計的特徴量
        if 'Age' in df.columns:
            df['Age_binned'] = pd.cut(
                df['Age'], 
                bins=[0, 12, 18, 60, 120],
                labels=['child', 'teen', 'adult', 'senior']
            )
            print(f"  - Age_binned生成（統計的）")
        
        # G23: 相互作用特徴量
        if 'Fare' in df.columns and 'FamilySize' in df.columns:
            df['Fare_per_person'] = df['Fare'] / df['FamilySize']
            print(f"  - Fare_per_person生成（相互作用）")
        
        print(f"✅ 特徴量エンジニアリング完了: {len(df.columns)}カラム")
        return df


class ModelTrainer:
    """
    G3: モデル構築
    
    対応するKAOS Goal:
    - G31: ベースラインモデル
    - G32: ハイパーパラメータ最適化
    - G33: アンサンブル（今回は省略）
    """
    
    def __init__(self, validator: AlloyConstraintValidator):
        self.validator = validator
        self.model = None
        self.hyperparameters = {}
    
    def set_hyperparameters(self, **kwargs):
        """
        G32: ハイパーパラメータ設定
        
        Alloy constraint: fact HyperparameterConstraints
        """
        print("\n[G32] ハイパーパラメータ設定...")
        self.hyperparameters = kwargs
        
        # Alloy制約検証
        self.validator.validate_hyperparameters(self.hyperparameters)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """
        G31: モデル訓練
        """
        print("\n[G31] モデル訓練開始...")
        
        self.model = RandomForestClassifier(**self.hyperparameters, random_state=42)
        self.model.fit(X, y)
        
        print(f"✅ RandomForest訓練完了")
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測実行
        
        Alloy constraint: fact PredictionBinary
        """
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        predictions = self.model.predict(X)
        
        # Alloy制約検証
        self.validator.validate_predictions_binary(predictions)
        
        return predictions


class ModelEvaluator:
    """
    G4: 評価・改善
    
    対応するKAOS Goal:
    - G41: クロスバリデーション
    - G42: リーダーボード確認（手動）
    - G43: 改善サイクル
    """
    
    def __init__(self, validator: AlloyConstraintValidator):
        self.validator = validator
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        G41: モデル評価
        
        Alloy constraint: fact EvaluationMetricsRange
        """
        print("\n[G41] モデル評価開始...")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Alloy制約検証
        self.validator.validate_metrics_range(metrics)
        
        # 結果表示
        for metric_name, value in metrics.items():
            print(f"  - {metric_name}: {value:.4f}")
        
        return metrics
    
    def cross_validate(self, model, X, y, cv=5) -> Dict[str, float]:
        """
        G41: クロスバリデーション
        """
        print(f"\n[G41] {cv}分割クロスバリデーション開始...")
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        result = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"  - CV平均: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
        
        return result


def main():
    """
    完全なKaggleパイプライン
    
    対応するAlloy predicate: CompleteKagglePipeline
    """
    print("="*60)
    print("Kaggle AI-Augmented形式手法パイプライン")
    print("="*60)
    
    # 初期化
    validator = AlloyConstraintValidator()
    
    # サンプルデータ生成（実際にはCSVから読み込み）
    print("\n[データ生成] サンプルデータを作成...")
    np.random.seed(42)
    
    train_df = pd.DataFrame({
        'PassengerId': range(1, 101),
        'Survived': np.random.randint(0, 2, 100),
        'Pclass': np.random.choice([1, 2, 3], 100),
        'Age': np.random.normal(30, 15, 100),
        'SibSp': np.random.poisson(0.5, 100),
        'Parch': np.random.poisson(0.3, 100),
        'Fare': np.random.exponential(30, 100),
        'Embarked': np.random.choice(['C', 'Q', 'S'], 100)
    })
    
    # 意図的に欠損値を作成
    train_df.loc[np.random.choice(100, 10), 'Age'] = np.nan
    train_df.loc[np.random.choice(100, 5), 'Embarked'] = np.nan
    
    test_df = train_df.copy()
    test_df = test_df.drop('Survived', axis=1)
    test_df['PassengerId'] = range(101, 201)
    
    print(f"  - 訓練データ: {len(train_df)}行")
    print(f"  - テストデータ: {len(test_df)}行")
    
    # Alloy制約: 訓練・テスト分離
    validator.validate_train_test_separation(
        set(train_df['PassengerId']),
        set(test_df['PassengerId'])
    )
    
    # G1: データ前処理
    preprocessor = DataPreprocessor(validator)
    train_df = preprocessor.handle_missing_values(train_df)
    train_df = preprocessor.handle_outliers(train_df)
    
    test_df = preprocessor.handle_missing_values(test_df)
    test_df = preprocessor.handle_outliers(test_df)
    
    # G2: 特徴量エンジニアリング
    fe = FeatureEngineer()
    train_df = fe.create_features(train_df)
    test_df = fe.create_features(test_df)
    
    # 特徴量選択
    feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'Fare_per_person']
    X_train = train_df[feature_cols]
    y_train = train_df['Survived']
    X_test = test_df[feature_cols]
    
    # G3: モデル構築
    trainer = ModelTrainer(validator)
    trainer.set_hyperparameters(
        max_depth=10,
        n_estimators=100,
        min_samples_split=5
    )
    model = trainer.train(X_train, y_train)
    
    # G4: 評価
    evaluator = ModelEvaluator(validator)
    
    # 訓練データでの評価
    train_predictions = trainer.predict(X_train)
    train_metrics = evaluator.evaluate(y_train, train_predictions)
    
    # クロスバリデーション
    cv_results = evaluator.cross_validate(model, X_train, y_train, cv=5)
    
    # テストデータでの予測
    test_predictions = trainer.predict(X_test)
    
    # 最終検証
    print("\n" + "="*60)
    print("最終Alloy制約検証")
    print("="*60)
    validator.validate_no_missing(train_df, "最終訓練データ")
    validator.validate_no_missing(test_df, "最終テストデータ")
    validator.validate_predictions_binary(test_predictions)
    
    print("\n" + "="*60)
    print("✅ すべてのAlloy制約を満たしました！")
    print("✅ CompleteKagglePipeline達成")
    print("="*60)
    
    # 結果サマリー
    print("\n[結果サマリー]")
    print(f"  訓練精度: {train_metrics['accuracy']:.4f}")
    print(f"  CV精度: {cv_results['cv_mean']:.4f}")
    print(f"  テスト予測数: {len(test_predictions)}")
    print(f"  予測分布: 0={sum(test_predictions==0)}, 1={sum(test_predictions==1)}")
    
    return train_df, test_df, model, test_predictions


if __name__ == '__main__':
    train_df, test_df, model, predictions = main()
