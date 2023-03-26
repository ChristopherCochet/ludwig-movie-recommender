from ludwig.api import LudwigModel
import pandas

# df = pandas.read_csv('rotten_tomatoes.csv')
# model = LudwigModel(config='rotten_tomatoes.yaml')
# results = model.train(dataset=df)

model = LudwigModel.load('results/experiment_run_6/model')

predictions, _ = model.predict(dataset='rotten_tomatoes_test.csv')
print(predictions.head())