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
import os
# %%
from fractions import Fraction


class Attribute:
    '''
    Attribute represents a feature of a dataset and contains the types of
    values and the frequency of it
    '''

    def __init__(self):
        # stores types of values
        self.valueCount = 0

        # stores types of values and frequency 
        self.values = {}

    def addValue(self, value):
        # adding new values to dictionary
        if value in self.values:
            self.values[value] += 1
        else:
            self.values[value] = 1
            self.valueCount += 1

    def addZeroValue(self, value):
        # initializing values qith zero frequency
        if value not in self.values:
            self.values[value] = 0

    def getValues(self):
        # returning all types of values
        return self.values.keys()

    def getFrequency(self, attr):
        # return frequency of a value
        return self.values[attr]

    def getValFreq(self):
        # return the dictionary
        return self.values

    def getNumOfValues(self):
        # return number of types of values
        return self.valueCount

    def getTotalValues(self):
        # get total of frequency from all values
        total = 0
        for i in self.values:
            total += self.values[i]
        return total


class Classifications:
    '''
    Classifications hold the types of classifications and stores attributes
    based on the classification. It also holds the frequency of the
    classification and the whole attributes data.
    '''

    def __init__(self, number, totalData):
        # the number of attributes of the datasets
        self.numberOfAttributes = number

        # stores types of classifications and the corresponding attributes data
        self.classifications = {}

        # stores the total number of data in the datasets
        self.totalNumber = 0

        # stores the attributes data without binding it to a classification
        self.globalAttributes = []

        # initialize global attributes
        self.initGlobalAttributes(number, totalData)

        # stores the frequency of each classifcation
        self.size = {}

    def initGlobalAttributes(self, number, totalData):
        # create Attribute objects
        for _ in range(self.numberOfAttributes):
            self.globalAttributes.append(Attribute())

        # initialize all types of all attributes using all data
        for data in totalData:
            for i in range(len(data) - 1):
                self.globalAttributes[i].addZeroValue(data[i])

    def addNewClassification(self, classification):
        # adding new types of classification
        self.classifications[classification] = []
        self.size[classification] = 0
        for _ in range(self.numberOfAttributes):
            self.classifications[classification].append(Attribute())

    def addNewDatas(self, datas, classification):
        # adding new attribute data to each classification
        if classification not in self.classifications:
            self.addNewClassification(classification)

        # add total number of data and frequency of each classification
        self.totalNumber += 1
        self.size[classification] += 1
        for i in range(self.numberOfAttributes):
            cleanedData = datas[i].rstrip('\n')
            # ignoring missing data
            if cleanedData == '?':
                continue
            
            # adding frequency of the attributes to global and based on 
            # classification
            self.globalAttributes[i].addValue(cleanedData)
            self.classifications[classification][i].addValue(cleanedData)

    def getClassifications(self):
        # get all types of classifications
        return self.classifications

    def getTotalNumber(self):
        # get total number of data
        return self.totalNumber

    def getTotalNumberOfClassification(self, classification):
        # return the frequency of a classification
        return self.size[classification]

    def getGlobalAttributeData(self, index):
        # return the global attribute data 
        return self.globalAttributes[index]

    def getClassificationTypes(self):
        # return all types of classifications
        return self.classifications.keys()

    def getAttributeDataIf(self, classification, index):
        # return attribute data corresponding to a classification
        return self.classifications[classification][index]

    def getNumberOfAttributes(self):
        # get number of attributes
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
        # adding zero frequency to a type of an attribute should during
        # processing, a type of an attribute is not found for a
        # classification
        for i in range(self.numberOfAttributes):
            types = self.globalAttributes[i].getValues()
            for c in self.classifications:
                for t in types:
                    self.classifications[c][i].addZeroValue(t)


# %%
# This function should open a data file in csv, and transform it into a usable format
# def preprocess(filepath):
def preprocess(datas, totalData):
    classifications = None
    attrCount = 0

    for data in datas:
        # initializing classification
        if attrCount == 0:
            attrCount = len(data)
            classifications = Classifications(attrCount - 1, totalData)

        if len(data) != attrCount:
            raise Exception(
                'All data should have the same number of attributes. The line is: {}'.format(data))

        # adding new data
        classification = data[attrCount - 1].rstrip('\n')
        classifications.addNewDatas(data[:-1], classification)

    # fixing missing values of the classification
    classifications.fixClassifications()
    return classifications


# %%
class Learner:
    '''
    The Learner class contains the necessary probabilities and all the
    calculations required to implement a Naive Bayes Classifier
    '''

    def __init__(self):
        # Stores the probability of a classification
        self.classProbability = {}

        # Stores the probability of each types of attribute given a type of
        # classification
        # Map<classification, [numOfAttr]List<Map<attributeType, Fraction>>>
        self.attrProbabilityIf = {}

    def learn(self, classification):
        # Calculating the probability necessary for the classifier
        for t in classification.getClassificationTypes():
            # probability of a classification
            self.classProbability[t] = Fraction(
                classification.getTotalNumberOfClassification(t),
                classification.getTotalNumber())

            # calculate attribute probability for a given classification
            data = []
            for i in range(classification.getNumberOfAttributes()):
                attr = classification.getAttributeDataIf(t, i)
                data.append(self.calculateProbabilities(attr))

            self.attrProbabilityIf[t] = data

    def getProbabilityIf(self, classification, attr, val):
        # print("Classification: ", classification)
        # print("Attribute: ", attr)
        # print("Type: ", val)

        # returning a neutral value for a missing values
        if val == '?':
            return 1
        
        # returning a probability given certain classification for a 
        # certain attribute
        return self.attrProbabilityIf[classification][attr][val]

    def getClassificationProbability(self, classification):
        # get probability for a classification
        return self.classProbability[classification]

    def predict(self, data, classification):
        # predict a classification given certain datas

        # get all probabilities given a classification
        possibilities = self.getProbabilityGivenData(data)
        classification = ""
        currentProbability = 0

        # find a classification with the highest probability
        for (k, v) in possibilities.items():
            if v > currentProbability:
                classification = k
                currentProbability = v

        return classification

    def getProbabilityGivenData(self, data):
        # iterate through the data and calculate probability
        possibilities = {}
        for c in self.classProbability.keys():
            possibilities[c] = 1

            # multiplying probabilities
            for i in range(len(data)):
                possibilities[c] *= self.getProbabilityIf(c, i, data[i])
            possibilities[c] *= self.getClassificationProbability(c)

        return possibilities

    def calculateProbabilities(self, attribute):
        # calculate probabilities for each attribute given a classification
        dict = {}
        total = attribute.getTotalValues()
        flag = False

        # calculating the frequency and total frequency of attributes
        for a in attribute.getValues():
            value = attribute.getFrequency(a)
            dict[a] = (value, total)
            if value == 0 and total != 0:
                flag = True

        # should there be a zero frequency, do probabilistic smoothing
        if flag:
            dict = self.probabilisticSmoothing(
                dict, attribute.getNumOfValues())

        # converting the tuples into Fraction
        for (k, v) in dict.items():
            freq, totalFreq = v
            if totalFreq == 0:
                dict[k] = 0
                continue
            dict[k] = Fraction(freq, totalFreq)

        return dict

    def probabilisticSmoothing(self, probabilities, numOfValues):
        # probabilistic smoothing to handle zero frequency
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
    # calculate prediction based on the given attributes
    return learner.predict(attributes, classifications)


# %%
# This function should evaluate a set of predictions, in a supervised context
def evaluate(dataTest, dataTrain):
    score = 0
    total = len(dataTest)

    classifications = preprocess(dataTrain, dataTest + dataTrain)
    classifier = train(classifications)
    for data in dataTest:
        classLabel = predict(data[:-1], classifier, classifications)
        score += 1 if classLabel == data[-1].rstrip('\n') else 0

    return Fraction(score, total)


# %%
# This function should calculate the Information Gain of an attribute or a set of attribute, with respect to the class
def info_gain():
    return



# %%
import os
import random

def splitFiles(trainPercentage, filepath):
    f = open(filepath, 'r')
    files = []
    length = 0
    for line in f.readlines():
        files.append(line.split(','))
        length += 1

    f.close()
    trainLength = int(length * trainPercentage)
    
    random.shuffle(files)

    return files[:trainLength], files[trainLength:]


path = 'C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\NaiveBayesPrediction\\2019S1-proj1-data_dos\\'
files = []
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('.csv'):
            files.append(dirpath + filename)

f = "test.txt"
trainPercentage = 0.8
data = []
print(len(files))
for fp in files:
    trainData, testData = splitFiles(trainPercentage, fp)
    # classifications = preprocess(trainData)
    print(fp)
    # classifications.printClassifications()

    print("Score: " + str(float(evaluate(testData, trainData))))

# primary-tumor.csv has its class ordered, that's why files are shuffled

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
