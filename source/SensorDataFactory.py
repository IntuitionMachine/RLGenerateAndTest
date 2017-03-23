import json
from random import randint

class SensorDataFactory:
    def __init__(self, dataFileName='OscilateSensorMedium.json'):
        self.dataFile = open(dataFileName)
        linesToSwallow = randint(0, 500)
        for lines in range(linesToSwallow):
            self.dataFile.readline()
       #self.dataFile.seek(randint(0,250))
    def getObservation(self):
        obs = self.dataFile.readline()
        if (obs == ''):
            self.dataFile.seek(0)
            linesToSwallow = randint(0,500)
            for lines in range(linesToSwallow):
                self.dataFile.readline()

            obs = self.dataFile.readline()
        return json.loads(obs)

    def close(self):
        self.dataFile.close()