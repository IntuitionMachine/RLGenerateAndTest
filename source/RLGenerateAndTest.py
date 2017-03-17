from GVF import *
from PredictionUnit import *
import json

class RLGenerateAndTest:
    def __init__(self):
        self.maxPosition = 1023
        self.minPosition = 510
        self.numberOfGVFs = 20
        self.numberOfRealFeatures = 20
        self.numberOfNoisyFeatures = 20
        self.thresholdGVF = 0.7
        self.initGVFs()
        self.predictionUnit = PredictionUnit(self.numberOfGVFs)
        self.previousX = False
        
        
    def initGVFs(self):
        #initialize a bunch of random GVFs each using a different random bit and random timestep
        for i in range(self.numberOfGVFs):
            gvf = self.initRandomGVF()
            gvf.name = str(i)
            #etc
            
    def initRandomGVF(self):
        vectorLength = self.numberOfRealFeatures + self.numberOfNoisyFeatures
        #def __init__(self, featureVectorLength, alpha, isOffPolicy, name = "GVF name"):

        gvf = GVF(featureVectorLength = vectorLength, alpha = 0.1, isOffPolicy = False)
        return gvf
        
    def replaceWeakestGVFs(self,numberToReplace):
        """
        #TODO
        indexesToReplace = self.predictionUnit.weakestWeights(numberToReplace)
        for index in indexesToReplace:
            self.predictionUnit.resetWeight(index)
            self.gvf[index] = self.initRandomGVF()
        """
        
    def updateGVFs(self, previousX, X):
        for gvf in self.gvfs:
            gvf.learn(lastState = previousX, action = False, newState = X)
    
    def XForObservation(self, observation):
        """
        Moving right
        {"load": 0.0, "temperature": 35, "timestamp": 1489682716.323766, "voltage": 12.3, "position": 539, "speed": 0}
        {"load": -0.046875, "temperature": 35, "timestamp": 1489682716.357122, "voltage": 12.3, "position": 554, "speed": 136}        
        """

        #todo return the rep
        #create the first self.numberOfRealFeatures and tack on random bits numberOfNoisyFeatures in length
        X = numpy.zeros(self.numberOfRealFeatures + self.numberOfNoisyFeatures)
        position = observation['position']
        tileIndex = (position - self.minPosition) / (self.maxPosition - self.minPosition) * self.numberOfRealFeatures / 2
        isMovingLeft = True
        if observation['speed'] >=0:
            isMovingLeft = False
        if not isMovingLeft:
            tileIndex = tileIndex + self.numberOfRealFeatures/2
        
        X[tileIndex] = 1.0
        #Make the remaining bits noisy
        X[self.numberOfRealFeatures:] = numpy.random.randint(2, size = 20)
         
        return X
        
    def thresholdOutputFromGVFs(self, X):
        thresholdPredictions = []
        for gvf in self.gvfs:
            prediction = gvf.prediction(X)
            thresholdPrediction = 0
            if prediction >= self.thresholdGVF:
                thresholdPrediction = 1
            else:
                thresholdPrediction = 0
            thresholdPredictions.append(thresholdPrediction)
            
        return thresholdPredictions
        
    def runExperiment(self, observationFile = 'OscilateSensorData.json'):
        with open(observationFile) as filePointer:
            for line in filePointer:
                #learn each gvf
                observation = json.loads(line)
                X = self.XForObservation(observation)
                y = observation["speed"]
                
                if self.previousX:
                    self.updateGVFs(self.previousX, X)

                
                #Have the prediction Unit learn
                predictionUnitInput = self.thresholdOutputFromGVFs(previousX)
                self.predictionUnit.learn(predictionUnitInput, y)
                
                #Remove the bad performing GVFs
                self.replaceWeakestGVFs()

                self.previousX = X                

                
                
                
        
        
