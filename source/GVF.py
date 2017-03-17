import numpy

class GVF:
    def __init__(self, featureVectorLength, alpha, isOffPolicy, name = "GVF name"):
        #set up lambda, gamma, etc.
        self.name = name
        self.isOffPolicy = isOffPolicy
        self.numberOfFeatures = featureVectorLength
        self.weights = numpy.zeros(self.numberOfFeatures)
        self.eligibilityTrace = numpy.zeros(self.numberOfFeatures)
        self.gammaLast = 1
        self.alpha = alpha
        #self.alpha = (1.0 - 0.90) * alpha
        self.i = 1

    """
    gamma, cumulant, and policy functions can/should be overiden by the specific instantiation of the GVF based on the intended usage.
    """
    def gamma(self, state):
        return 0.0

    def cumulant(self, state):
        return 1

    def policy(self, state):
        #To be overwritten based on GVF's intended behavior if off policy. Otherwise 1 means on policy
        return 1

    def lam(self, state):
        return 0.90

    def learn(self, lastState, action, newState):
        if self.isOffPolicy:
            self.gtdLearn(lastState, action, newState)
        else:
            self.tdLearn(lastState, newState)

    def tdLearn(self, lastState, newState):
        print("!!!!! LEARN  !!!!!!!")
        print("GVF name: " + str(self.name))
        pred = self.prediction(lastState)
        print("--- Prediction for " + str(lastState) + ", " + str(lastState) + " before learning: " + str(pred))

        zNext = self.cumulant(newState)
        #print("Cumulant: " + str(zNext))
        gammaNext = self.gamma(newState)
        #print("gammaNext: " + str(gammaNext))
        lam = self.lam(newState)
        #print("gammaLast: " + str(self.gammaLast))

        #print("lambda: " + str(lam))
        self.eligibilityTrace = self.gammaLast * lam * self.eligibilityTrace + lastState

        tdError = zNext + gammaNext * numpy.inner(newState, self.weights) - numpy.inner(lastState, self.weights)

        #print("tdError: " + str(tdError))
        self.weights = self.weights + self.alpha * tdError * self.eligibilityTrace

        pred = self.prediction(lastState)
        print("Prediction for " + str(lastState)  + " after learning: " + str(pred))

        self.gammaLast = gammaNext

    def prediction(self, stateRepresentation):
        return numpy.inner(self.weights, stateRepresentation)
