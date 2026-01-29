# 特徴量追加の手順（5分で完了）

## 📍 修正場所

**ファイル**: `kaggle_alloy_implementation.py`  
**クラス**: `FeatureEngineer`  
**メソッド**: `create_features()`  
**行数**: 約212-254行目

---

## 🔧 修正方法

### ステップ1: 追加位置を確認

`kaggle_alloy_implementation.py`を開いて、以下の部分を探します：

```python
class FeatureEngineer:
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... 既存のコード ...
        
        # G23: 相互作用特徴量
        if 'Fare' in df.columns and 'FamilySize' in df.columns:
            df['Fare_per_person'] = df['Fare'] / df['FamilySize']
            self.new_features.append('Fare_per_person')
            print(f"  🔧 Fare_per_person生成（相互作用）")
        
        # 👇 ここに新しい特徴量を追加 👇
        
        # Alloy制約検証: 新特徴量が存在する
        self.validator.validate_new_features_exist(df, self.new_features)
```

### ステップ2: 新しい特徴量を追加

上記の「👇 ここに」の位置に、以下をコピー＆ペースト：

```python
        # G24: Name（敬称）から特徴量生成
        if 'Name' in df.columns:
            df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            # 敬称をグループ化
            title_mapping = {
                'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss',
                'Lady': 'Rare', 'Jonkheer': 'Rare', 'Don': 'Rare',
                'Dona': 'Rare', 'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
            }
            df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
            df['Title_encoded'] = pd.factorize(df['Title'])[0]
            self.new_features.extend(['Title_encoded'])
            print(f"  🔧 Title_encoded生成（ドメイン知識）")
        
        # G25: Cabin（客室）から特徴量生成
        if 'Cabin' in df.columns:
            df['Cabin_letter'] = df['Cabin'].str[0].fillna('U')
            df['Has_Cabin'] = df['Cabin'].notna().astype(int)
            df['Cabin_letter_encoded'] = pd.factorize(df['Cabin_letter'])[0]
            self.new_features.extend(['Has_Cabin', 'Cabin_letter_encoded'])
            print(f"  🔧 Has_Cabin, Cabin_letter_encoded生成（ドメイン知識）")
        
        # G26: Sex（性別）を数値化
        if 'Sex' in df.columns:
            df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
            self.new_features.append('Sex_encoded')
            print(f"  🔧 Sex_encoded生成（前処理）")
        
        # G27: Embarked（乗船港）を数値化
        if 'Embarked' in df.columns:
            df['Embarked_encoded'] = pd.factorize(df['Embarked'])[0]
            self.new_features.append('Embarked_encoded')
            print(f"  🔧 Embarked_encoded生成（前処理）")
```

### ステップ3: 特徴量選択を更新

同じファイルの`KagglePipeline.execute()`メソッド（約440行目）を探して：

```python
# 修正前
feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 
                'FamilySize', 'IsAlone', 'Fare_per_person']

# 修正後
feature_cols = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'IsAlone', 'Fare_per_person', 'Age_binned_numeric',
    'Title_encoded', 'Has_Cabin', 'Cabin_letter_encoded',
    'Sex_encoded', 'Embarked_encoded'
]
```

### ステップ4: 保存して実行

```bash
# 保存したら実行
python run_titanic_fixed.py
```

---

## ✅ 期待される結果

### 実行ログ

```
[G2] 特徴量エンジニアリング開始...
  🔧 FamilySize, IsAlone生成（ドメイン知識）
  🔧 Age_binned_numeric生成（統計的）
  🔧 Fare_per_person生成（相互作用）
  🔧 Title_encoded生成（ドメイン知識）      ← 新規
  🔧 Has_Cabin, Cabin_letter_encoded生成（ドメイン知識） ← 新規
  🔧 Sex_encoded生成（前処理）              ← 新規
  🔧 Embarked_encoded生成（前処理）         ← 新規

✅ [G2] 新特徴量検証合格: [...12個...]
```

### パフォーマンス改善

```
修正前:
  訓練精度: 0.8496
  CV精度: 0.7184

修正後（期待値）:
  訓練精度: 0.85-0.90
  CV精度: 0.75-0.80  ← 3-8%改善
```

---

## 🎯 なぜこの特徴量が効果的か

| 特徴量 | 理由 |
|--------|------|
| **Title** | 社会的地位（Mr, Mrs, Masterなど）は生存率に影響 |
| **Has_Cabin** | 客室あり＝上級客室＝生存率高い |
| **Cabin_letter** | デッキ階層（A-G）は沈没時の脱出に影響 |
| **Sex_encoded** | 女性優先の救命ボート |
| **Embarked** | 乗船港により客層が異なる |

---

## 📚 参考情報

### Alloy制約との関係

新しい特徴量も既存のAlloy制約を満たします：

```alloy
fact FeatureEngineeringRules {
  all fed: FeatureEngineeredData |
    some fed.newFeatures
}
```

`self.new_features`リストに追加することで、この制約が自動的に検証されます。

### さらに精度を上げるには

```python
# Ageグループ×Pclassの相互作用
if 'Age_binned_numeric' in df.columns and 'Pclass' in df.columns:
    df['Age_Pclass'] = df['Age_binned_numeric'] * 10 + df['Pclass']
    self.new_features.append('Age_Pclass')

# 家族サイズのカテゴリ化
if 'FamilySize' in df.columns:
    df['FamilySize_category'] = pd.cut(df['FamilySize'], 
                                       bins=[0, 1, 4, 20],
                                       labels=[0, 1, 2]).astype(int)
    self.new_features.append('FamilySize_category')
```

---

## ⚠️ トラブルシューティング

### エラー1: KeyError

```
KeyError: 'Title_encoded'
```

**原因**: 特徴量選択リストに追加したが、生成されていない

**解決**: `self.new_features.append('Title_encoded')`の行を確認

### エラー2: Alloy制約違反

```
[G2違反] 特徴量 Title_encoded が生成されていません
```

**原因**: `self.new_features`への登録を忘れた

**解決**: 必ず`self.new_features.append()`または`.extend()`を実行

---

**作成日**: 2026年1月28日  
**所要時間**: 5分  
**期待される改善**: CV精度 +3-8%
