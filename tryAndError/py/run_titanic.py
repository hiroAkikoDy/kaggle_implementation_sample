# run_titanic.py を作成
import pandas as pd
from kaggle_alloy_implementation import KagglePipeline

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

pipeline = KagglePipeline()
predictions, results = pipeline.execute(train_df, test_df)

# submission.csv 作成
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)