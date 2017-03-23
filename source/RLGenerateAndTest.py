from GVF import *
from PredictionUnit import *
from SensorDataFactory import *
import json
import matplotlib.pyplot as plt


def testLearning(numTests):
    rl = RLGenerateAndTest()
    gvf = GVF(featureVectorLength = 40, alpha = 0.01, isOffPolicy = False)
    cumulantFunction = rl.makeVectorBitCumulantFunction(0)
    gvf.cumulant = cumulantFunction    
    
    for i in range(numTests):
        xOld = rl.XForObservation({"load": -0.046875, "temperature": 35, "timestamp": 1489682716.357122, "voltage": 12.3, "position": 554, "speed": 136}) 
        xNew = rl.XForObservation({"load": -0.046875, "temperature": 35, "timestamp": 1489682716.357122, "voltage": 12.3, "position": 554, "speed": -136})
        #print("Prediction before: " + str(gvf.prediction(xOld)))
        gvf.learn(xOld, xNew)
        
        xOld2 = rl.XForObservation({"load": -0.046875, "temperature": 35, "timestamp": 1489682716.357122, "voltage": 12.3, "position": 900, "speed": 136}) 
        xNew2 = rl.XForObservation({"load": -0.046875, "temperature": 35, "timestamp": 1489682716.357122, "voltage": 12.3, "position": 1020, "speed": 136}) 
        gvf.learn(xOld2, xNew2)
                
        xOld3 = rl.XForObservation({"load": -0.046875, "temperature": 35, "timestamp": 1489682716.357122, "voltage": 12.3, "position": 712, "speed": 136}) 
        xNew3 = rl.XForObservation({"load": -0.046875, "temperature": 35, "timestamp": 1489682716.357122, "voltage": 12.3, "position": 750, "speed": 136})        
        gvf.learn(xOld3, xNew3)
        
        #print("Prediction after learning: " + str(gvf.prediction(xOld)))
    return gvf
        
class RLGenerateAndTest:
    def __init__(self):
        self.observationFactory = SensorDataFactory()
        self.maxPosition = 1023.0
        self.minPosition = 510.0
        self.numberOfRealFeatures = 20
        self.numberOfNoisyFeatures = 20
        self.candidateBits = list(range(self.numberOfNoisyFeatures + self.numberOfRealFeatures))
        #self.numberOfNoisyFeatures = 0
        #self.numberOfGVFs = self.numberOfRealFeatures + self.numberOfNoisyFeatures
        self.numberOfGVFs = self.numberOfRealFeatures
        self.gvfThreshold = 0.65
        self.gvfs = self.initGVFs()
        self.predictionUnit = PredictionUnit(self.numberOfGVFs)
        self.previousValue = False
        self.previousX = numpy.zeros(self.numberOfRealFeatures + self.numberOfNoisyFeatures)
        
    def resetForRun(self):
        self.gvfs = self.initGVFs()
        self.predictionUnit = PredictionUnit(self.numberOfGVFs)
        self.previousValue = False
        self.previousX = numpy.zeros(self.numberOfRealFeatures + self.numberOfNoisyFeatures)
        self.candidateBits = list(range(self.numberOfNoisyFeatures + self.numberOfRealFeatures))

    def initGVFs(self):
        #initialize a bunch of random GVFs each using a different random bit and random timestep
        gvfs = []
        for i in range(self.numberOfGVFs):
            gvf = self.initRandomGVF(excludeBitsTried = True)
            gvfs.append(gvf)
        return gvfs
            
    def makeVectorBitCumulantFunction(self, bitIndex):
        def cumulantFunction(X):
            if (X[bitIndex] == 1):
                return 1
            else:
                return 0
        return cumulantFunction

    def randomBitIndex(self, excludeBitsTried = False):
        randomBit = 0
        if not excludeBitsTried:
            randomBit = numpy.random.randint(self.numberOfRealFeatures + self.numberOfNoisyFeatures)
        else:
            #Naive strategy. If there are no candidate bits left to choose from, repopulate them all
            if (len(self.candidateBits) == 0):
                self.candidateBits = list(range(self.numberOfNoisyFeatures + self.numberOfRealFeatures))
            randomBit = numpy.random.choice(self.candidateBits)

        return randomBit
        #exclusions is an array of indexes to not choose from


    def initRandomGVF(self, excludeBitsTried = False):
        vectorLength = self.numberOfRealFeatures + self.numberOfNoisyFeatures
        #TODO swap comments after testing
        gvf = GVF(featureVectorLength = vectorLength, alpha = 0.1 / (self.numberOfNoisyFeatures*0.5), isOffPolicy = False)

        #gvf = GVF(featureVectorLength = vectorLength, alpha = 0.1, isOffPolicy = False)

        randomBit = self.randomBitIndex(excludeBitsTried)
        if randomBit in self.candidateBits:
            self.candidateBits.remove(randomBit)
        cumulantFunction = self.makeVectorBitCumulantFunction(randomBit)
        gvf.cumulant = cumulantFunction
        gvf.name = "Cumulant bit: " + str(randomBit)

        return gvf
        
    def replaceWeakestGVFs(self,numberToReplace):
        if len(self.candidateBits) == 0:
            print("---- Not replacing any GVFS since all options exhausted")
        else:
            indexesToReplace = self.predictionUnit.weakestWeights(numberToReplace)
            for index in indexesToReplace:
                print("---- Replacing " + self.gvfs[index].name  +" ----")
            print("---- Replacing: ")
            for index in indexesToReplace:
                self.predictionUnit.resetWeight(index)
                self.gvfs[index] = self.initRandomGVF(excludeBitsTried = True)
                print("---- New GVF: " + self.gvfs[index].name)
        
    def updateGVFs(self, previousX, X):
        for gvf in self.gvfs:
            gvf.learn(lastState = previousX, newState = X)

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

        tileIndex = int(self.numberOfRealFeatures * 0.5 * ((position - self.minPosition) / (self.maxPosition - self.minPosition)))
        isMovingLeft = True
        if observation['speed'] >=0:
            isMovingLeft = False
        if not isMovingLeft:
            tileIndex = tileIndex + int(self.numberOfRealFeatures/2)
        
        X[tileIndex] = 1
        #Make the remaining bits noisy
        #TODO - Add back in randomizing after testing
        #X[self.numberOfRealFeatures:] = numpy.random.randint(2, size = 20)
        #70% chance of being a 0
        X[self.numberOfRealFeatures:] = numpy.random.choice(a=[0, 1], p=[0.8, 0.2], size = self.numberOfNoisyFeatures)
        return X
        
    def thresholdOutputFromGVFs(self, X):
        thresholdOutputs = numpy.zeros(len(self.gvfs))
        for i in range(len(self.gvfs)):
            gvf = self.gvfs[i]
            #print("GVF Name: " + str(gvf.name))
            #print("Current State: " + str(X))
            prediction = gvf.prediction(X)
            #print("Prediction: " + str(prediction))
            thresholdPrediction = 0
            if prediction >= self.gvfThreshold:
                thresholdPrediction = 1
            else:
                thresholdPrediction = 0
            thresholdOutputs[i] = thresholdPrediction
            #print("Threshold Prediction: " + str(thresholdPrediction))
        return thresholdOutputs

    def runExperiment(self, numberOfRuns=1, numberOfObservations=10000):
        averageObservationErrors = numpy.zeros(numberOfObservations) # -1 because there's no error after the first obs.
        observationErrors = numpy.zeros(numberOfObservations)
        for run in range(numberOfRuns):
            print("+++++++++ Run number " + str(run) + "++++++++++++")
            observationErrors = numpy.zeros(numberOfObservations)
            cullingCount = 0
            for observationNumber in range(numberOfObservations):
                if (observationNumber % 5000 == 0):
                    print("======== Run " + str(run) + ", Observation " + str(observationNumber) + " =======")
                observation = self.observationFactory.getObservation()
                X = self.XForObservation(observation)
                # print(X)
                y = observation["speed"]

                if self.previousValue:
                    self.updateGVFs(self.previousX, X)
                    # Have the prediction Unit learn
                    predictionUnitInput = self.thresholdOutputFromGVFs(X)
                    self.predictionUnit.learn(predictionUnitInput, y)
                    prediction = self.predictionUnit.prediction(predictionUnitInput)
                    error = y - prediction

                    observationErrors[observationNumber] = error
                    """
                    if X[4]==1:
                        observationErrors[observationNumber] = error
                    else:
                        observationErrors[observationNumber] = 0
                    """

                else:
                    self.previousValue = True
                self.previousX = X

                #Cull and replace
                numberToReplace = 1
                if((observationNumber+1) % 5000  == 0):
                    if cullingCount == 0:
                        numberToReplace = 4
                    self.replaceWeakestGVFs(numberToReplace)
                    #print("!! Replacing " + str(numberToReplace) + " GVFs")
                    cullingCount = cullingCount + 1


            averageObservationErrors = averageObservationErrors + (1 / (run + 1)) * observationErrors
            self.resetForRun()

        return averageObservationErrors

    def plotAverageError(self, averageErrors):
        fig = plt.figure(1)
        fig.suptitle('Prediction Error', fontsize=14, fontweight='bold')
        ax = fig.add_subplot(211)
        titleLabel = "Generate And Test"
        ax.set_title(titleLabel)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Average Error')

        ax.plot(averageErrors)

        #ax.plot(optimalActionsNonStationary)

        plt.show()


    def runOldExperiment(self, observationFile = 'OscilateSensorDataLarge.json'):
        for i in range(30):
            print("+++++++ Run number "+ str(i) + " +++++++")
            with open(observationFile) as filePointer:
                sampleNumber = 0
                for line in filePointer:
                    sampleNumber = sampleNumber + 1
                    if sampleNumber % 13000 == 0:
                        print("Learning " + str(1) + " ...")
                    if (sampleNumber %100 == 0):
                        print("100")
                    #learn each gvf
                    observation = json.loads(line)
                    X = self.XForObservation(observation)
                    #print(X)
                    y = observation["speed"]

                    if self.previousValue:
                        self.updateGVFs(self.previousX, X)
                        #Have the prediction Unit learn
                        #print("")
                        """
                        activeBit = 0
                        for bit in range(20):
                            if X[bit]==1:
                                activeBit = bit
                        """
                        """
                        for gvf in self.gvfs:
                            print("Rupee: " + str(gvf.rupee()))
                        """
                        #print("============== Observation: " + str(sampleNumber) + ", run: " + str(i) + " ===================")
                        #print("X: " + str(X) + ", active bit: " + str(activeBit) +  ", y: " + str(y))
                        #print("Active bit: " + str(activeBit) + ", y: " + str(y))
                        predictionUnitInput = self.thresholdOutputFromGVFs(X)
                        #if 1 in predictionUnitInput:
                        if (1 in predictionUnitInput):
                            print("Found an activation")
                        if (numpy.count_nonzero(predictionUnitInput) > 1):
                            print("============== Observation: " + str(sampleNumber) + ", run: " + str(
                                i) + " ===================")
                            print(predictionUnitInput)
                        self.predictionUnit.learn(predictionUnitInput, y)
                        predictionAfter = self.predictionUnit.prediction(predictionUnitInput)
                        #print("Prediction: " + str(predictionAfter))
                        #print("Prediction input: " + str(predictionUnitInput))
                        #for gvf in self.gvfs:
                        #    print(" - name: " + str(gvf.name))
                        #    print(" - prediction: " + str(gvf.prediction(X)))
                        #Remove the bad performing GVFs
                    else:
                        self.previousValue= True
                    self.previousX = X
            numberToReplace = 1
            if (i == 0):
                numberToReplace = 5
            self.replaceWeakestGVFs(numberToReplace)
        print("Done")

"""
rl = RLGenerateAndTest()
avgErrors = rl.runExperiment(numberOfRuns = 2, numberOfObservations = 100000)
rl.plotAverageError(avgErrors)
"""

#testLearning(10)

                
                
        
        
