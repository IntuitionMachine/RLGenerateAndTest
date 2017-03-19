import numpy

class PredictionUnit:
    def __init__(self, inputLength, alpha = 0.05):
        #initialize all of the weights
        self.inputLength = inputLength
        self.weights = numpy.zeros(inputLength)
        self.age = numpy.zeros(inputLength)
        self.alpha = alpha
    
    def learn(self, X, y):
        print("PredictionUnit.learn()")
        #y is the actual value to learn from
        #print("y: " + str(y))
        #print("X: " + str(X))
        #print("self.weights: " + str(self.weights))
        tdError = y - self.prediction(X)
        #print("tdError: " + str(tdError))
        self.weights = self.weights + self.alpha*tdError*X
        addOneVector = [1.0] * self.inputLength
        self.age = self.age + addOneVector

    def prediction(self, X):
        p = numpy.inner(self.weights, X)
        return p
        
    def weakestWeights(self, numberOfWeights):
        indexes = numpy.argpartition(self.weights, numberOfWeights)
        return indexes[:numberOfWeights]
        
    def resetWeight(index):
        self.weights[index] = 0.0
        self.age[index] = 0.0
        