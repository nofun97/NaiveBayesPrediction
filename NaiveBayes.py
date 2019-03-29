# %% [markdown]
#  # The University of Melbourne, School of Computing and Information Systems
#  # COMP30027 Machine Learning, 2019 Semester 1
#  -----
#  ## Project 1: Gaining Information about Naive Bayes
#  -----
#  ###### Student Name(s): Novan
#  ###### Python version: 3.7
#  ###### Submission deadline: 1pm, Fri 5 Apr 2019
# %% [markdown]
#  This iPython notebook is a template which you may use for your Project 1 submission. (You are not required to use it; in particular, there is no need to use iPython if you do not like it.)
#
#  Marking will be applied on the five functions that are defined in this notebook, and to your responses to the questions at the end of this notebook.
#
#  You may change the prototypes of these functions, and you may write other functions, according to your requirements. We would appreciate it if the required functions were prominent/easy to find.

# %%
from fractions import Fraction


class Attribute:

    def __init__(self):
        self.valueCount = 0
        self.values = {}

    def addValue(self, value):
        if value in self.values:
            self.values[value] += 1
        else:
            self.values[value] = 1
            self.valueCount += 1

    def addZeroValue(self, value):
        if value not in self.values:
            self.values[value] = 0

    def getValues(self):
        return self.values.keys()

    def getFrequency(self, attr):
        return self.values[attr]

    def getValFreq(self):
        return self.values

    def getNumOfValues(self):
        return self.valueCount

    def getTotalValues(self):
        total = 0
        for i in self.values:
            total += self.values[i]
        # print("ALSO HERE: " + str(self.values))
        return total


class Classifications:

    def __init__(self, number, totalData):
        self.numberOfAttributes = number
        self.classifications = {}
        self.totalNumber = 0
        self.globalAttributes = []
        self.initGlobalAttributes(number, totalData)
        self.size = {}

    def initGlobalAttributes(self, number, totalData):
        for _ in range(self.numberOfAttributes):
            self.globalAttributes.append(Attribute())

        for data in totalData:
            for i in range(len(data) - 1):
                self.globalAttributes[i].addZeroValue(data[i])

    def addNewClassification(self, classification):
        self.classifications[classification] = []
        self.size[classification] = 1
        for _ in range(self.numberOfAttributes):
            self.classifications[classification].append(Attribute())

    def addNewDatas(self, datas, classification):
        if classification not in self.classifications:
            self.addNewClassification(classification)

        self.totalNumber += 1
        self.size[classification] += 1
        for i in range(self.numberOfAttributes):
            cleanedData = datas[i].rstrip('\n')
            if cleanedData == '?':
                continue
            self.globalAttributes[i].addValue(cleanedData)
            self.classifications[classification][i].addValue(cleanedData)

    def getGlobalAttributeProbability(self, data):
        probability = 1
        for i in range(len(data)):
            if data[i] == '?':
                continue

            attr = self.globalAttributes[i]
            freq = attr.getFrequency(data[i])
            if freq == 0:
                continue
            probability *= Fraction(attr.getFrequency(
                data[i]), attr.getTotalValues())
        return probability

    def getClassifications(self):
        return self.classifications

    def getTotalNumber(self):
        return self.totalNumber

    def getTotalNumberOfClassification(self, classification):
        c = self.classifications[classification]
        total = 0
        for i in c:
            total += i.getTotalValues()

        return total

    def getGlobalAttributeData(self, index):
        return self.globalAttributes[index]

    def getClassificationTypes(self):
        return self.classifications.keys()

    def getAttributeDataIf(self, classification, index):
        return self.classifications[classification][index]

    def getNumberOfAttributes(self):
        return self.numberOfAttributes

    def printClassifications(self):
        print("Global values")
        for i in self.globalAttributes:
            print(i.getValFreq())

        print('\n')
        for c in self.classifications:
            print("Classification: " + c)
            for a in self.classifications[c]:
                print(a.getValFreq())

            print('\n')

    def writeClassifications(self, filename):
        file = open(filename, "w+")
        file.write("Global values\n")
        for i in self.globalAttributes:
            file.write(str(i.getValFreq()) + '\n')

        file.write('\n')
        for c in self.classifications:
            file.write("Classification: " + c + '\n')
            for a in self.classifications[c]:
                file.write(str(a.getValFreq()) + '\n')

            file.write('\n')

        file.close()

    def fixClassifications(self):
        for i in range(self.numberOfAttributes):
            types = self.globalAttributes[i].getValues()
            for c in self.classifications:
                for t in types:
                    self.getAttributeDataIf(c, i).addZeroValue(t)


# %%
# This function should open a data file in csv, and transform it into a usable format
# def preprocess(filepath):
#     f = open(filepath, 'r')
#     classifications = None
#     attr_count = 0

#     for line in f.readlines():
#         vals = line.split(',')

#         if attr_count == 0:
#             attr_count = len(vals)
#             classifications = Classifications(attr_count - 1)

#         if len(vals) != attr_count:
#             raise Exception(
#                 'All data should have the same number of attributes. The line is: {}'.format(line))

#         classification = vals[attr_count - 1].rstrip('\n')
#         classifications.addNewDatas(vals[:-1], classification)

#     f.close()
#     classifications.fixClassifications()
#     return classifications


def preprocess(datas, totalData):
    classifications = None
    attr_count = 0

    for data in datas:
        if attr_count == 0:
            attr_count = len(data)
            classifications = Classifications(attr_count - 1, totalData)

        if len(data) != attr_count:
            raise Exception(
                'All data should have the same number of attributes. The line is: {}'.format(data))

        classification = data[attr_count - 1].rstrip('\n')
        classifications.addNewDatas(data[:-1], classification)

    classifications.fixClassifications()
    return classifications


# %%
class Learner:

    def __init__(self):
        # Map<classification, Fraction>
        self.classProbability = {}

        # Map<classification, [numOfAttr]List<Map<attributeType, Fraction>>>
        self.attrProbabilityIf = {}

    def learn(self, classification):
        for t in classification.getClassificationTypes():
            self.classProbability[t] = Fraction(
                classification.getTotalNumberOfClassification(t),
                classification.getTotalNumber())

            data = []
            for i in range(classification.getNumberOfAttributes()):
                attr = classification.getAttributeDataIf(t, i)
                data.append(self.calculateProbabilities(attr))

            self.attrProbabilityIf[t] = data
        # print(self.attrProbabilityIf)
        # print(self.classProbability)
        # print(self.)
    # def printProbabilities(self):
    #     for (c, p) in self.classProbability:
    #         print(str(c) + ": " + str(p))

    #     for (c, a) in self.attrProbabilityIf:

    def getProbabilityIf(self, classification, attr, val):
        #         print("HERE")
        #         print(self.attrProbabilityIf)
        # print("Classification: ", classification)
        # print("Attribute: ", attr)
        # print("Type: ", val)
        if val == '?':
            return 1
        return self.attrProbabilityIf[classification][attr][val]

    def getClassificationProbability(self, classification):
        return self.classProbability[classification]

    def predict(self, data, classification):
        possibilities = self.getProbabilityGivenData(data)
        attributesProbability = classification.getGlobalAttributeProbability(
            data)
        classification = ""
        currentProbability = 0

        for (k, v) in possibilities.items():
            p = Fraction(v, attributesProbability)
            if p > currentProbability:
                classification = k
                currentProbability = p

        return classification

    def getProbabilityGivenData(self, data):
        possibilities = {}
        for c in self.classProbability:
            possibilities[c] = 1
            for i in range(len(data)):
                possibilities[c] *= self.getProbabilityIf(c, i, data[i])
            possibilities[c] *= self.getClassificationProbability(c)
        # print(possibilities)
        return possibilities

    def calculateProbabilities(self, attribute):
        dict = {}
        total = attribute.getTotalValues()
        flag = False
        for a in attribute.getValues():
            value = attribute.getFrequency(a)
            dict[a] = (value, total)
            if value == 0 and total != 0:
                flag = True

        if flag:
            dict = self.probabilisticSmoothing(
                dict, attribute.getNumOfValues())

        # print("HERE: " + str(dict))
        for (k, v) in dict.items():
            freq, totalFreq = v
            if totalFreq == 0:
                dict[k] = Fraction(freq, 1)
                continue
            dict[k] = Fraction(freq, totalFreq)

        return dict

    def probabilisticSmoothing(self, probabilities, numOfValues):
        dict = {}
        for (k, v) in probabilities.items():
            freq, totalFreq = v
            dict[k] = (1 + freq, numOfValues + totalFreq)

        return dict


# %%
# This function should build a supervised NB model
def train(classifications):
    learner = Learner()
    learner.learn(classifications)

    return learner


# %%
# This function should predict the class for an instance or a set of instances, based on a trained model
def predict(attributes, learner, classifications):
    return learner.predict(attributes, classifications)


# %%
# This function should evaluate a set of predictions, in a supervised context
def evaluate(dataTest, dataTrain):
    score = 0
    total = len(dataTest)

    classifications = preprocess(dataTrain, dataTest + dataTrain)
    # classifications.printClassifications()
    classifier = train(classifications)
    for data in dataTest:
        classLabel = predict(data[:-1], classifier, classifications)
        # print("Guess: {} Real: {} Judge: {}".format(classLabel, data[-1], classLabel == data[-1]))
        score += 1 if classLabel == data[-1].rstrip('\n') else 0

    # print("Score: {} Total: {}".format(score, total))
    return Fraction(score, total)


# %%
# This function should calculate the Information Gain of an attribute or a set of attribute, with respect to the class
def info_gain():
    return



# %%
import os


def splitFiles(trainPercentage, filepath):
    f = open(filepath, 'r')
    files = []
    length = 0
    for line in f.readlines():
        files.append(line.split(','))
        length += 1

    f.close()
    trainLength = int(length * trainPercentage)

    return files[:trainLength], files[trainLength:]


path = './2019S1-proj1-data_dos/'
files = []
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('.csv'):
            files.append(dirpath + filename)

f = "test.txt"
trainPercentage = 0.5
data = []
# files.append(path + 'anneal.csv')
for fp in files:
    trainData, testData = splitFiles(trainPercentage, fp)
    # classifications = preprocess(trainData)
    print(fp)
    # classifications.printClassifications()

    print("Score: " + str(float(evaluate(testData, trainData))))

# %% [markdown]
#  Questions (you may respond in a cell or cells below):
#
#  1. The Naive Bayes classifiers can be seen to vary, in terms of their effectiveness on the given datasets (e.g. in terms of Accuracy). Consider the Information Gain of each attribute, relative to the class distribution — does this help to explain the classifiers’ behaviour? Identify any results that are particularly surprising, and explain why they occur.
#  2. The Information Gain can be seen as a kind of correlation coefficient between a pair of attributes: when the gain is low, the attribute values are uncorrelated; when the gain is high, the attribute values are correlated. In supervised ML, we typically calculate the Infomation Gain between a single attribute and the class, but it can be calculated for any pair of attributes. Using the pair-wise IG as a proxy for attribute interdependence, in which cases are our NB assumptions violated? Describe any evidence (or indeed, lack of evidence) that this is has some effect on the effectiveness of the NB classifier.
#  3. Since we have gone to all of the effort of calculating Infomation Gain, we might as well use that as a criterion for building a “Decision Stump” (1-R classifier). How does the effectiveness of this classifier compare to Naive Bayes? Identify one or more cases where the effectiveness is notably different, and explain why.
#  4. Evaluating the model on the same data that we use to train the model is considered to be a major mistake in Machine Learning. Implement a hold–out or cross–validation evaluation strategy. How does your estimate of effectiveness change, compared to testing on the training data? Explain why. (The result might surprise you!)
#  5. Implement one of the advanced smoothing regimes (add-k, Good-Turing). Does changing the smoothing regime (or indeed, not smoothing at all) affect the effectiveness of the Naive Bayes classifier? Explain why, or why not.
#  6. Naive Bayes is said to elegantly handle missing attribute values. For the datasets with missing values, is there any evidence that the performance is different on the instances with missing values, compared to the instances where all of the values are present? Does it matter which, or how many values are missing? Would a imputation strategy have any effect on this?
#
#  Don't forget that groups of 1 student should respond to question (1), and one other question of your choosing. Groups of 2 students should respond to question (1) and question (2), and two other questions of your choosing. Your responses should be about 150-250 words each.
