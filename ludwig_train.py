from ludwig.api import LudwigModel
import pandas

df = pandas.read_csv('rotten_tomatoes.csv')
model = LudwigModel(config='rotten_tomatoes.yaml')
results = model.train(dataset=df)