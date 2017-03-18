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

    def lam(self, state):
        return 0.90

    def learn(self, lastState, newState):
        self.tdLearn(lastState, newState)

    def tdLearn(self, lastState, newState):

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

        self.gammaLast = gammaNext

    def prediction(self, stateRepresentation):
        return numpy.inner(self.weights, stateRepresentation)
