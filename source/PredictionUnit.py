import numpy

class PredictionUnit:
    def__init__(inputLength, alpha = 0.01):
        #initialize all of the weights
        self.inputLength = inputLength
        self.weights = numpy.zeros(inputLength)
        self.age = numpy.zeros(inputLength)
        self.alpha = alpha
    
    def learn(self, X, y)
        #y is the actual value to learn from
        tdError = y - self.prediction(X)
        self.weights = self.weights + self.alpha*tdError
        addOneVector = [1.0] * self.inputLength
        self.age = self.age + addOneVector

    def prediction(self, X)
        p = numpy.inner(self.weights, X)
        return
        
    def weakestWeights(self, numberOfWeights):
        indexes = numpy.zeros(numberOfWeights)
        #go over the indexes and return the indexes of the weakest ones. 
        return indexes
        
    def resetWeight(index):
        self.weights[index] = 0.0
        self.age[index] = 0.0
        