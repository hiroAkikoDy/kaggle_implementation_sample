"""
Kaggle AI-Augmented形式手法 - 拡張版（特徴量追加）

Alloyモデル: kaggle_competition_v3_final.als
検証結果:
  ✅ G1_Achieved: データ品質保証
  ✅ G2_Achieved: 特徴量エンジニアリング（拡張版）
  ✅ G3_Achieved: モデル構築
  ⚠️ G4_Achieved: 実装レベルで追加（Alloy検証なし）

対象: タイタニック生存予測
追加特徴量: Title, Cabin, Sex, Embarked
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AlloyConstraintValidator:
    """
    Alloy形式記法で定義された制約をPythonで検証
    
    対応するAlloyモデル: kaggle_competition_v3_final.als
    検証済みゴール: G1, G2, G3
    """
    
    @staticmethod
    def validate_no_missing(df: pd.DataFrame, stage: str = "") -> bool:
        """
        Alloy: fact MissingValueHandling
        Alloy: fact NumericalMissingValues
        Alloy: fact CategoricalMissingValues
        
        制約: 前処理後のデータには欠損値がない
        """
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            raise ValueError(f"[G1違反] {stage}: 欠損値が{missing_count}個残っています")
        print(f"✅ [G1] {stage}: 欠損値チェック合格")
        return True
    
    @staticmethod
    def validate_outliers(df: pd.DataFrame, column: str, min_val: float, max_val: float) -> bool:
        """
        Alloy: fact OutlierConstraints
        制約: 値が妥当な範囲内（例: 年齢は0-120）
        """
        if column not in df.columns:
            return True
            
        actual_min = df[column].min()
        actual_max = df[column].max()
        
        if actual_min < min_val or actual_max > max_val:
            raise ValueError(
                f"[G1違反] {column}の範囲制約違反: "
                f"期待[{min_val}, {max_val}], 実際[{actual_min}, {actual_max}]"
            )
        print(f"✅ [G1] {column}: 外れ値制約合格 [{actual_min:.2f}, {actual_max:.2f}]")
        return True
    
    @staticmethod
    def validate_hyperparameters(params: Dict) -> bool:
        """
        Alloy: fact HyperparameterConstraints
        制約: ハイパーパラメータが妥当な範囲
        """
        if 'max_depth' in params:
            depth = params['max_depth']
            if not (1 <= depth <= 20):
                raise ValueError(f"[G3違反] max_depthは[1, 20]の範囲: {depth}")
        
        if 'n_estimators' in params:
            n_est = params['n_estimators']
            if not (1 <= n_est <= 1000):
                raise ValueError(f"[G3違反] n_estimatorsは[1, 1000]の範囲: {n_est}")
        
        print(f"✅ [G3] ハイパーパラメータ検証合格: {params}")
        return True
    
    @staticmethod
    def validate_predictions_binary(predictions: np.ndarray) -> bool:
        """
        Alloy: fact PredictionBinary
        制約: 予測値は0または1
        """
        unique_values = set(predictions)
        if not unique_values.issubset({0, 1}):
            raise ValueError(f"[G3違反] 予測値は0または1である必要があります: {unique_values}")
        print(f"✅ [G3] 予測値バイナリチェック合格")
        return True
    
    @staticmethod
    def validate_new_features_exist(df: pd.DataFrame, expected_features: List[str]) -> bool:
        """
        Alloy: fact FeatureEngineeringRules (simplified)
        制約: 新しい特徴量が存在する
        """
        for feature in expected_features:
            if feature not in df.columns:
                raise ValueError(f"[G2違反] 特徴量 {feature} が生成されていません")
        print(f"✅ [G2] 新特徴量検証合格: {expected_features}")
        return True
    
    @staticmethod
    def validate_train_test_separation(train_indices: set, test_indices: set) -> bool:
        """
        Alloy: fact TrainTestSeparation
        制約: 訓練データとテストデータは重複しない
        """
        overlap = train_indices & test_indices
        if len(overlap) > 0:
            raise ValueError(f"[G1違反] 訓練とテストが{len(overlap)}行重複しています")
        print(f"✅ [G1] 訓練・テスト分離チェック合格")
        return True


class DataPreprocessor:
    """
    G1: データ品質保証
    
    対応するAlloy述語: G1_Achieved
    対応するKAOS Goal:
    - G11: 欠損値処理
    - G12: 外れ値処理
    - G13: データ型整合性
    """
    
    def __init__(self, validator: AlloyConstraintValidator):
        self.validator = validator
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        G11: 欠損値処理
        
        Alloy constraints:
        - fact NumericalMissingValues
        - fact CategoricalMissingValues
        """
        print("\n" + "="*60)
        print("[G11] 欠損値処理開始...")
        print("="*60)
        
        df = df.copy()
        
        # 数値特徴量（中央値で補完）
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  📊 {col}: 中央値{median_val:.2f}で補完")
        
        # カテゴリカル特徴量（最頻値で補完）
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"  📊 {col}: 最頻値'{mode_val}'で補完")
        
        # Alloy制約検証
        self.validator.validate_no_missing(df, "欠損値処理後")
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        G12: 外れ値処理
        
        Alloy constraint: fact OutlierConstraints
        """
        print("\n" + "="*60)
        print("[G12] 外れ値処理開始...")
        print("="*60)
        
        df = df.copy()
        
        # 年齢: 0-120歳の範囲に制限
        if 'Age' in df.columns:
            df['Age'] = df['Age'].clip(0, 120)
            self.validator.validate_outliers(df, 'Age', 0, 120)
        
        # 運賃: 0以上
        if 'Fare' in df.columns:
            df['Fare'] = df['Fare'].clip(0, None)
            print(f"✅ [G12] Fare: 負の値を0にクリップ")
        
        return df


class FeatureEngineer:
    """
    G2: 特徴量生成（拡張版）
    
    対応するAlloy述語: G2_Achieved
    対応するKAOS Goal:
    - G21: ドメイン知識特徴量
    - G22: 統計的特徴量
    - G23: 相互作用特徴量
    - G24-G27: 追加特徴量（拡張版）
    """
    
    def __init__(self, validator: AlloyConstraintValidator):
        self.validator = validator
        self.new_features = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alloy constraint: fact FeatureEngineeringRules (simplified)
        新特徴量は存在する（具体的な生成方法は実装レベルで決定）
        
        拡張版: より多くの特徴量を生成
        """
        print("\n" + "="*60)
        print("[G2] 特徴量エンジニアリング開始（拡張版）...")
        print("="*60)
        
        df = df.copy()
        
        # G21: ドメイン知識に基づく特徴量
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            self.new_features.extend(['FamilySize', 'IsAlone'])
            print(f"  🔧 FamilySize, IsAlone生成（ドメイン知識）")
        
        # G22: 統計的特徴量
        if 'Age' in df.columns:
            df['Age_binned'] = pd.cut(
                df['Age'], 
                bins=[0, 12, 18, 60, 120],
                labels=['child', 'teen', 'adult', 'senior']
            )
            # カテゴリカルを数値に変換
            df['Age_binned_numeric'] = df['Age_binned'].cat.codes
            # カテゴリカル版は削除（数値版のみ使用）
            df = df.drop('Age_binned', axis=1)
            self.new_features.append('Age_binned_numeric')
            print(f"  🔧 Age_binned_numeric生成（統計的）")
        
        # G23: 相互作用特徴量
        if 'Fare' in df.columns and 'FamilySize' in df.columns:
            df['Fare_per_person'] = df['Fare'] / df['FamilySize']
            self.new_features.append('Fare_per_person')
            print(f"  🔧 Fare_per_person生成（相互作用）")
        
        # ========================================
        # 👇 拡張版：新しい特徴量を追加 👇
        # ========================================
        
        # G24: Name（敬称）から特徴量生成【新規】
        if 'Name' in df.columns:
            # 敬称を抽出（Mr., Mrs., Miss. など）
            df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            
            # 敬称をグループ化（希少な敬称をまとめる）
            title_mapping = {
                'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss',
                'Lady': 'Rare', 'Jonkheer': 'Rare', 'Don': 'Rare',
                'Dona': 'Rare', 'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
            }
            df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
            
            # 敬称を数値化
            df['Title_encoded'] = pd.factorize(df['Title'])[0]
            self.new_features.append('Title_encoded')
            print(f"  🔧 Title_encoded生成（ドメイン知識・新規）")
        
        # G25: Cabin（客室）から特徴量生成【新規】
        if 'Cabin' in df.columns:
            # Cabinの最初の文字（デッキ階層: A, B, C, D, E, F, G）
            df['Cabin_letter'] = df['Cabin'].str[0].fillna('U')
            
            # Cabinがあるかどうか（生存率に影響）
            df['Has_Cabin'] = df['Cabin'].notna().astype(int)
            
            # Cabin_letterを数値化
            df['Cabin_letter_encoded'] = pd.factorize(df['Cabin_letter'])[0]
            
            self.new_features.extend(['Has_Cabin', 'Cabin_letter_encoded'])
            print(f"  🔧 Has_Cabin, Cabin_letter_encoded生成（ドメイン知識・新規）")
        
        # G26: Sex（性別）を数値化【新規】
        if 'Sex' in df.columns:
            df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
            self.new_features.append('Sex_encoded')
            print(f"  🔧 Sex_encoded生成（前処理・新規）")
        
        # G27: Embarked（乗船港）を数値化【新規】
        if 'Embarked' in df.columns:
            df['Embarked_encoded'] = pd.factorize(df['Embarked'])[0]
            self.new_features.append('Embarked_encoded')
            print(f"  🔧 Embarked_encoded生成（前処理・新規）")
        
        # ========================================
        # 👆 ここまで新しい特徴量 👆
        # ========================================
        
        # Alloy制約検証: 新特徴量が存在する
        self.validator.validate_new_features_exist(df, self.new_features)
        
        print(f"\n✅ [G2] 特徴量エンジニアリング完了: {len(df.columns)}カラム")
        print(f"  📊 新規生成特徴量: {len(self.new_features)}個")
        return df


class ModelTrainer:
    """
    G3: モデル構築
    
    対応するAlloy述語: G3_Achieved
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
        print("\n" + "="*60)
        print("[G32] ハイパーパラメータ設定...")
        print("="*60)
        
        self.hyperparameters = kwargs
        
        # Alloy制約検証
        self.validator.validate_hyperparameters(self.hyperparameters)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """
        G31: モデル訓練
        """
        print("\n" + "="*60)
        print("[G31] モデル訓練開始...")
        print("="*60)
        
        self.model = RandomForestClassifier(**self.hyperparameters, random_state=42)
        self.model.fit(X, y)
        
        print(f"✅ [G31] RandomForest訓練完了")
        print(f"  📊 特徴量数: {X.shape[1]}")
        print(f"  📊 訓練データ数: {X.shape[0]}")
        
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
    G4: 評価・改善（実装レベルのみ、Alloy検証なし）
    
    注意: G4はAlloyで検証されていません
    実装レベルで追加された機能です
    """
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        モデル評価
        
        注意: Alloy検証なし（G4は形式化困難のため）
        """
        print("\n" + "="*60)
        print("[G4] モデル評価開始（Alloy検証なし）...")
        print("="*60)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # 結果表示
        for metric_name, value in metrics.items():
            print(f"  📊 {metric_name}: {value:.4f}")
        
        return metrics
    
    def cross_validate(self, model, X, y, cv=5) -> Dict[str, float]:
        """
        クロスバリデーション
        """
        print(f"\n[G4] {cv}分割クロスバリデーション開始...")
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        result = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"  📊 CV平均: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
        
        return result


class KagglePipeline:
    """
    完全なKaggleパイプライン（G1-G3: Alloy検証済み、G4: 実装のみ）
    拡張版：より多くの特徴量を使用
    """
    
    def __init__(self):
        self.validator = AlloyConstraintValidator()
        self.preprocessor = DataPreprocessor(self.validator)
        self.feature_engineer = FeatureEngineer(self.validator)
        self.trainer = ModelTrainer(self.validator)
        self.evaluator = ModelEvaluator()
    
    def execute(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                target_col: str = 'Survived') -> Tuple[np.ndarray, Dict]:
        """
        完全なパイプライン実行
        
        Alloy述語: PracticalKagglePipeline (G1 ∧ G2 ∧ G3)
        """
        print("\n" + "="*70)
        print("🚀 Kaggle AI-Augmented形式手法パイプライン実行（拡張版）")
        print("="*70)
        print(f"Alloyモデル: kaggle_competition_v3_final.als")
        print(f"検証済みゴール: G1, G2, G3")
        print(f"実装レベル: G4")
        print(f"バージョン: 特徴量拡張版")
        print("="*70)
        
        # 訓練・テスト分離検証
        self.validator.validate_train_test_separation(
            set(train_df.index),
            set(test_df.index)
        )
        
        # ターゲット保存
        y_train = train_df[target_col]
        train_df = train_df.drop(target_col, axis=1)
        
        # G1: データ前処理
        print("\n" + "🔵 "*35)
        print("ステージ1: G1 - データ品質保証（Alloy検証済み）")
        print("🔵 "*35)
        
        train_df = self.preprocessor.handle_missing_values(train_df)
        train_df = self.preprocessor.handle_outliers(train_df)
        
        test_df = self.preprocessor.handle_missing_values(test_df)
        test_df = self.preprocessor.handle_outliers(test_df)
        
        # G2: 特徴量エンジニアリング
        print("\n" + "🟢 "*35)
        print("ステージ2: G2 - 特徴量エンジニアリング（Alloy検証済み・拡張版）")
        print("🟢 "*35)
        
        train_df = self.feature_engineer.create_features(train_df)
        test_df_fe = FeatureEngineer(self.validator)
        test_df = test_df_fe.create_features(test_df)
        
        # 特徴量選択（拡張版）
        feature_cols = [
            # 基本特徴量
            'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            # 既存の生成特徴量
            'FamilySize', 'IsAlone', 'Fare_per_person', 'Age_binned_numeric',
            # 新規追加特徴量
            'Title_encoded', 'Has_Cabin', 'Cabin_letter_encoded',
            'Sex_encoded', 'Embarked_encoded'
        ]
        
        # 存在する特徴量のみ選択
        feature_cols = [col for col in feature_cols if col in train_df.columns]
        
        print(f"\n📊 使用特徴量: {len(feature_cols)}個")
        print(f"  {', '.join(feature_cols)}")
        
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        
        # G3: モデル構築
        print("\n" + "🟡 "*35)
        print("ステージ3: G3 - モデル構築（Alloy検証済み）")
        print("🟡 "*35)
        
        self.trainer.set_hyperparameters(
            max_depth=10,
            n_estimators=100,
            min_samples_split=5
        )
        model = self.trainer.train(X_train, y_train)
        
        # G4: 評価（実装レベル）
        print("\n" + "🟣 "*35)
        print("ステージ4: G4 - 評価（実装レベル、Alloy検証なし）")
        print("🟣 "*35)
        
        # 訓練データでの評価
        train_predictions = self.trainer.predict(X_train)
        train_metrics = self.evaluator.evaluate(y_train, train_predictions)
        
        # クロスバリデーション
        cv_results = self.evaluator.cross_validate(model, X_train, y_train, cv=5)
        
        # テストデータでの予測
        test_predictions = self.trainer.predict(X_test)
        
        # 最終検証
        print("\n" + "="*70)
        print("🔍 最終Alloy制約検証（G1-G3）")
        print("="*70)
        
        self.validator.validate_no_missing(train_df, "最終訓練データ")
        self.validator.validate_no_missing(test_df, "最終テストデータ")
        self.validator.validate_predictions_binary(test_predictions)
        
        print("\n" + "="*70)
        print("✅ すべてのAlloy制約（G1-G3）を満たしました！")
        print("✅ PracticalKagglePipeline達成")
        print("="*70)
        
        # 結果サマリー
        print("\n" + "📊 "*35)
        print("結果サマリー（拡張版）")
        print("📊 "*35)
        print(f"  訓練精度: {train_metrics['accuracy']:.4f}")
        print(f"  CV精度: {cv_results['cv_mean']:.4f}")
        print(f"  テスト予測数: {len(test_predictions)}")
        print(f"  予測分布: 0={sum(test_predictions==0)}, 1={sum(test_predictions==1)}")
        
        return test_predictions, {
            'train_metrics': train_metrics,
            'cv_results': cv_results
        }


def main():
    """
    サンプルデータでパイプライン実行
    """
    print("="*70)
    print("Kaggle AI-Augmented形式手法 - サンプル実行（拡張版）")
    print("="*70)
    
    # サンプルデータ生成
    print("\n[データ生成] サンプルデータを作成...")
    np.random.seed(42)
    
    n_train = 100
    n_test = 50
    
    train_df = pd.DataFrame({
        'PassengerId': range(1, n_train + 1),
        'Survived': np.random.randint(0, 2, n_train),
        'Pclass': np.random.choice([1, 2, 3], n_train),
        'Age': np.random.normal(30, 15, n_train),
        'SibSp': np.random.poisson(0.5, n_train),
        'Parch': np.random.poisson(0.3, n_train),
        'Fare': np.random.exponential(30, n_train),
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_train)
    })
    
    # 意図的に欠損値を作成
    train_df.loc[np.random.choice(n_train, 10, replace=False), 'Age'] = np.nan
    train_df.loc[np.random.choice(n_train, 5, replace=False), 'Embarked'] = np.nan
    
    test_df = pd.DataFrame({
        'PassengerId': range(n_train + 1, n_train + n_test + 1),
        'Pclass': np.random.choice([1, 2, 3], n_test),
        'Age': np.random.normal(30, 15, n_test),
        'SibSp': np.random.poisson(0.5, n_test),
        'Parch': np.random.poisson(0.3, n_test),
        'Fare': np.random.exponential(30, n_test),
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_test)
    }, index=range(n_train, n_train + n_test))
    
    # 欠損値を作成
    age_missing_indices = test_df.index[np.random.choice(n_test, 5, replace=False)]
    test_df.loc[age_missing_indices, 'Age'] = np.nan
    
    print(f"  ✅ 訓練データ: {len(train_df)}行")
    print(f"  ✅ テストデータ: {len(test_df)}行")
    
    # パイプライン実行
    pipeline = KagglePipeline()
    predictions, results = pipeline.execute(train_df, test_df)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    
    print("\n" + "="*70)
    print("🎉 パイプライン完了（拡張版）")
    print("="*70)
    print(f"提出ファイル: {len(submission)}行")
    print(f"\n{submission.head(10)}")
    
    return submission, results


if __name__ == '__main__':
    submission, results = main()
