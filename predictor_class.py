import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#predictor class
#training_data = [np.array(training), np.array(labels)], lyrs = [layers.], raw_data - indicators(dataframe)
class predictor:
    def __init__(self, model_type='standard', lrs=None,  optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.MeanSquaredError(), scope=1, *args, **kwargs): #it's a regression model so no accuracy here
        
        self.optimizer = optimizer #
        self.loss = loss
        self.scope = scope
        self.lrs = lrs
        if model_type == 'standard':
          self.model = tf.keras.Sequential()
          for each in self.lrs: self.model.add(each)
          self.model.compile(optimizer = self.optimizer, loss = self.loss)
          
        else:
          self.model = model_type #not a str definetely

    def train(self, training_data, labels, epochs=100):
       self.model.fit(x=training_data, y=labels, epochs=epochs, shuffle=True)

    def examine_bias(self, raw_data, training_data, labels):
      predicts = [] 
      biases = []
      for i in range(len(labels)-1):
        prediction = self.pred(training_data[i:i+1])
        predicts.append(prediction)
        biases.append((labels[i] - prediction)/prediction)

      predicts = np.array(predicts)
      predicts = np.append(np.array([None]*(len(raw_data) - len(training_data) + 1)), np.reshape(predicts, (predicts.shape[0], )))
        
      positive = [i for i in biases if i < 0]
      negative = [i for i in biases if i > 0]
      
      for each in [positive, negative]:
        if len(each) == 0:
          each.append(0)

      avg_positive = sum(positive)/len(positive)
      avg_negative = sum(negative)/len(negative)
      self.bias = (avg_positive, avg_negative) #estimation of how pessimistic/optimistic the model is
    
    def pred(self, data):
      return self.model.predict(data, verbose=0)
    
    def make_prediction(self, data):
      p = self.pred(data) 
      return (p+p*self.bias[0], p, p+p*self.bias[1])
    
    def see_performance(self, ins, training_data): #dataframe and array of data like after createdataset
      ps = [] 
      for i in range(len(training_data)-1):
        p = self.make_prediction(training_data[i:i+1])
        ps.append(np.reshape(p, (len(p))))
      
      plt.plot(ps)
      plt.plot(ins.close.values)
      plt.show()
    

## methods for data engineering
def calcMACD(data):  #this counts the key statistical indicator for the strategy. MACD in my case
    prices = data['Close']
    indicator = prices.ewm(span=12, adjust=False, min_periods=12).mean() - prices.ewm(span=26, adjust=False, min_periods=26).mean()
    signal = indicator.ewm(span=9, adjust=False, min_periods=9).mean()
    d = indicator - signal
    return d

def ma(data, span):
  mean = []
  for e in range(len(data[span:])):
    mean.append(sum(data[e-span:e])/span)
  return np.array(mean)

def createdataset(secu):
  indicators = pd.DataFrame([])
  indicators['open'], indicators['close'], indicators['high'], indicators['low'] = secu.Open[100:], secu.Close[100:], secu.High[100:], secu.Low[100:]
  indicators['macdhist'] = calcMACD(secu)[74:]
  indicators['ma20'], indicators['ma50'] = ma(secu.Close, 20)[80:], ma(secu.Close, 50)[50:]
  return indicators

def get_trainingdata(indicators, scope=1):
  training = []
  labels = []
  training_full = []
  for i in range(0, len(indicators)-5-scope):
    ins = pd.DataFrame([indicators[i:i+5].open,
                        indicators[i:i+5].close,
                        indicators[i:i+5].high,
                        indicators[i:i+5].low,
                        indicators[i:i+5].macdhist,
                        indicators[i:i+5].ma20,
                        indicators[i:i+5].ma50])

    y = indicators.close[i+5+scope-1]
    pic = np.reshape(ins.values, (5, 7, 1))

    training.append(pic)
    labels.append(y)
    training_full.append((pic, y))

  training = np.array(training)
  labels = np.array(labels)
  return training, labels

#pipeline of creating a predictor object: obj = predictor(lyrs)
#                                         train on a variety of securities
#                                         obj.examine_bias(on a security that the estimator is purposed for)
#                                         ready for making predictions