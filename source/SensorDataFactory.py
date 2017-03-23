class SensorDataFactory:
    def __init__(self, dataFileName):
        self.dataFile = open(dataFileName)

    def getObservation(self):
        obs = self.dataFile.readline()
        if (obs == ''):
            self.dataFile.seek(0)
            obs = self.dataFile.readline()
        return obs

    def close(self):
        self.dataFile.close()