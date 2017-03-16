import TileCoder
import GVF
import PredictionUnit
import json

class RLGenerateAndTest:
    def __init__(self):
        self.numberOfGVFS = 20
        self.numberOfRealFeatures = 20
        self.numberOfNoisyFeatures = 20
        self.thresholdGVF = 0.7
        self.initGVFs()
        self.predictionUnit = PredictionUnit(self.numberOfGVFs)
        self.previousX = False
        
        
    def initGVFs():
        #initialize a bunch of random GVFs each using a different random bit and random timestep
        for i in range(self.numberOfGVFS):
            gvf = self.initRandomGVF()
            #etc
            
    def initRandomGVF(self):
        vectorLength = self.numberOfRealFeatures + self.numberOfNoisyFeatures
        gvf = GVF(vectorLength)
        return gvf
        
    def replaceWeakestGVFs(numberToReplace):
        """
        #TODO
        indexesToReplace = self.predictionUnit.weakestWeights(numberToReplace)
        for index in indexesToReplace:
            self.predictionUnit.resetWeight(index)
            self.gvf[index] = self.initRandomGVF()
        """
        
    def updateGVFs(self, previousX, X):
        for gvf in self.gvfs:
            gvf.learn(previousX, X)
    
    def XForObservation(observation):
        #todo return the rep
        #create the first self.numberOfRealFeatures and tack on random bits numberOfNoisyFeatures in length
        X = numpy.zeros(self.numberOfRealFeatures + self.numberOfNoisyFeatures)
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
        open(observationFile) as filePointer:
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

                
                
                
        
        
