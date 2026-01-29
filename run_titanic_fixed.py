"""
タイタニックコンペ実行スクリプト（修正版）

修正内容：
- インデックスをPassengerIdに設定して、訓練・テスト分離エラーを回避
- Alloy制約 fact TrainTestSeparation を満たす
"""
import pandas as pd
from kaggle_alloy_implementation import KagglePipeline


def main():
    print("="*70)
    print("Kaggle タイタニックコンペ実行")
    print("="*70)
    
    # データ読み込み
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"\n📂 訓練データ: {len(train_df)}行")
    print(f"📂 テストデータ: {len(test_df)}行")
    
    # 🔧 修正: インデックスをPassengerIdに設定
    # 
    # 問題: Kaggleのtrain.csvとtest.csvは、デフォルトで
    #       インデックスが0から始まるため重複してしまう
    #       → Alloy制約 fact TrainTestSeparation 違反
    # 
    # 解決: PassengerIdをインデックスに設定
    #       train: PassengerId 1-891
    #       test:  PassengerId 892-1309
    #       → 重複なし！
    
    # PassengerIdを保存（提出ファイル用）
    test_passenger_ids = test_df['PassengerId'].copy()
    
    # インデックス設定
    train_df = train_df.set_index('PassengerId')
    test_df = test_df.set_index('PassengerId')
    
    print(f"\n✅ インデックス設定完了")
    print(f"  訓練: {train_df.index.min()} - {train_df.index.max()}")
    print(f"  テスト: {test_df.index.min()} - {test_df.index.max()}")
    print(f"  重複チェック: {len(set(train_df.index) & set(test_df.index))}個")
    
    # パイプライン実行
    pipeline = KagglePipeline()
    predictions, results = pipeline.execute(train_df, test_df)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Survived': predictions
    })
    submission.to_csv('submission.csv', index=False)
    
    # 結果表示
    print("\n" + "="*70)
    print("🎉 パイプライン完了")
    print("="*70)
    print(f"✅ submission.csv作成完了！")
    print(f"\n📊 結果:")
    print(f"  訓練精度: {results['train_metrics']['accuracy']:.4f}")
    print(f"  訓練精度(詳細): precision={results['train_metrics']['precision']:.4f}, "
          f"recall={results['train_metrics']['recall']:.4f}, "
          f"f1={results['train_metrics']['f1_score']:.4f}")
    print(f"  CV精度: {results['cv_results']['cv_mean']:.4f} "
          f"(±{results['cv_results']['cv_std']:.4f})")
    print(f"\n📊 予測:")
    print(f"  テスト予測数: {len(predictions)}")
    print(f"  生存予測: {sum(predictions==1)}人")
    print(f"  死亡予測: {sum(predictions==0)}人")
    print(f"  生存率: {sum(predictions==1)/len(predictions)*100:.1f}%")
    
    print(f"\n📄 提出ファイルの最初の10行:")
    print(submission.head(10))
    
    print(f"\n🚀 次のステップ:")
    print(f"  kaggle competitions submit -c titanic -f submission.csv -m \"Alloy検証済み\"")
    
    return submission, results


if __name__ == '__main__':
    submission, results = main()
