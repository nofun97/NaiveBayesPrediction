# %% [markdown]
#  # The University of Melbourne, School of Computing and Information Systems
#  # COMP30027 Machine Learning, 2019 Semester 1
#  -----
#  ## Project 1: Gaining Information about Naive Bayes
#  -----
#  ###### Student Name(s): Novan
#  ###### Python version: 3.6
#  ###### Submission deadline: 1pm, Fri 5 Apr 2019
# %% [markdown]
#  This iPython notebook is a template which you may use for your Project 1 submission. (You are not required to use it; in particular, there is no need to use iPython if you do not like it.)
#
#  Marking will be applied on the five functions that are defined in this notebook, and to your responses to the questions at the end of this notebook.
#
#  You may change the prototypes of these functions, and you may write other functions, according to your requirements. We would appreciate it if the required functions were prominent/easy to find.

# %%
# necessary imports
import os
import random
from fractions import Fraction
from math import log

# %%
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
        if value == '?':
            return
        # adding new values to dictionary
        if value in self.values:
            self.values[value] += 1
        else:
            self.values[value] = 1
            self.valueCount += 1

    def addZeroValue(self, value):
        # initializing values qith zero frequency
        if value not in self.values and value != '?':
            self.values[value] = 0

    def getValues(self):
        # returning all types of values
        return self.values.keys()

    def getFrequency(self, attr):
        # return frequency of a value
        if attr not in self.values:
            return 0
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
        if classification == '?':
            return
        # adding new types of classification
        self.classifications[classification] = []
        self.size[classification] = 0
        for _ in range(self.numberOfAttributes):
            self.classifications[classification].append(Attribute())

    def addNewDatas(self, datas, classification):
        if classification == '?':
            return
        # adding new attribute data to each classification
        if classification not in self.classifications:
            self.addNewClassification(classification)

        # add total number of data and frequency of each classification

        self.totalNumber += 1 if classification != '?' else 0
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

    def fixClassifications(self):
        # adding zero frequency to a type of an attribute should during
        # processing, a type of an attribute is not found for a
        # classification
        for i in range(self.numberOfAttributes):
            types = self.globalAttributes[i].getValues()
            for c in self.classifications:
                for t in types:
                    self.classifications[c][i].addZeroValue(t)

    def calculateClassEntropy(self):
        entropy = 0
        for c in self.getClassificationTypes():
            # using frequency of a class divided by total number of data
            freqRatio = Fraction(self.size[c], self.getTotalNumber())
            entropy += freqRatio * log(freqRatio, 2)
        return -1 * entropy

    def calculateTotalFreqAttrType(self, attrIndex, attrType):
        # Calculating total frequency of a certain attribute value
        total = 0

        # adding all the frequency of an attribute type from each class
        for c in self.getClassificationTypes():
            attr = self.getAttributeDataIf(c, attrIndex)
            total += attr.getFrequency(attrType)

        return total

    def calculateAttrValueEntropy(self, attrIndex, attrType):
        entropy = 0
        totalAttrType = self.calculateTotalFreqAttrType(attrIndex, attrType)

        # ignoring attribute with zero frequency
        if totalAttrType == 0:
            return 0

        # calculating for each classification
        for c in self.getClassificationTypes():
            freq = self.getAttributeDataIf(c, attrIndex).getFrequency(attrType)
            if freq == 0:
                continue

            # frequency of a class divided by total attribute type frequency
            valFreqRatio = Fraction(freq, totalAttrType)
            entropy += valFreqRatio * log(valFreqRatio, 2)

        return -1 * entropy

    def calculateMeanInfo(self, attrIndex):
        globalAttr = self.getGlobalAttributeData(attrIndex)
        totalFreq = self.getTotalNumber()
        meanInfo = 0

        # calculating for each value of an attribute
        for t in globalAttr.getValues():
            freq = self.calculateTotalFreqAttrType(attrIndex, t)

            # frequency of an attribute type divided by total number of data
            freqRatio = Fraction(freq, totalFreq)
            entropy = self.calculateAttrValueEntropy(attrIndex, t)
            meanInfo += freqRatio * entropy

        return meanInfo

    def calculateInfoGain(self):
        infoGain = []
        classEntropy = self.calculateClassEntropy()

        # for each attribute, calculate information gain
        for i in range(self.getNumberOfAttributes()):
            mi = self.calculateMeanInfo(i)
            infoGain.append(classEntropy - mi)
        return infoGain

# %%
# This function should open a data file in csv, and transform it into a usable format

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
        classification = data[-1].rstrip('\n')
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
def info_gain(classifications):
    return classifications.calculateInfoGain()
# %%
# script to run evaluation on all the datasets

def prepFiles(filepath):
    # creates a list of instances from a file
    f = open(filepath, 'r')
    files = []
    length = 0
    for line in f.readlines():
        files.append(line.split(','))
        length += 1

    f.close()
    return files


def collectFiles(mainDir):
    # walk through directory and get all the csv files
    files = []
    for (dirpath, dirnames, filenames) in os.walk(mainDir):
        for filename in filenames:
            if filename.endswith('.csv'):
                files.append(dirpath + filename)
    return files

# getting the filepaths in a directory
files = collectFiles(
    "/Users/novan/Desktop/CODE/Machine Learning/assignment1/2019S1-proj1-data/"
    )
for fp in files:
    # printing the filepath, scores, and info gain for each file
    data = prepFiles(fp)
    classifications = preprocess(data, data)
    print("Filepath: " + fp)
    print("Score: ", float(evaluate(data, data)))
    print("Info Gain: ", info_gain(classifications))
    print("\n")

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

# %% [markdown]
# Scores (Using training data as test data)

# Filename: hypothyroid.csv
# Score:  0.9516282010749288
# Info Gain:  [0.004628873031652547, 0.0009139351160850073, 0.0012382074503017315, 0.00014844815831743796, 0.0009985293906336068, 0.0013683791752741592, 0.0005423006444424394, 0.0004350938464638965, 0.0004888757691284829, 0.0008983004044028076, 4.463778824304043e-05, 7.8684698479492e-05, 0.009353710215580346, 0.004075493419623766, 0.005792553705846859, 0.005768288201614624, 0.005744031245602799, 0.002580427555574416]


# Filename: primary-tumor.csv
# Score:  0.4365781710914454
# Info Gain:  [0.1547421418870596, 0.33536005150555503, 1.0262234265467285, 2.0947714990558133, 0.21246189904816637, 0.0203669388480483, 0.10088123982399111, 0.0678727757044233, 0.22052193470670511, 0.1997614363902529, 0.06714460241010656, 0.06025390884525317, 0.29153013602249356, 0.12715354518198252, 0.2458886814337724, 0.18425767171538476, 0.17014811083887338]


# Filename: hepatitis.csv
# Score:  0.832258064516129
# Info Gain:  [0.03660746514280977, 0.015265380561918285, 0.014490701150154384, 0.08645063847884216, 0.08322845589007444, 0.013806029835453981, 0.0903522944652434, 0.09049078626122975, 0.058739302370822144, 0.12938741279822152, 0.15163520023638288, 0.10012174391687245, 0.08493296456638777]


# Filename: anneal.csv
# Score:  0.8129175946547884
# Info Gain:  [0.40908953764451006, 0.0, 0.3060515354289405, 0.051344088764404106, 0.29108220585994726, 0.1471188622809556, 0.2137228803159087, 0.29223544065798446, 0.1261663361036096, 0.14107379163812883, 0.032488406491841815, 0.43517783626288575, 0.03870173274881061, 0.00043760652021185287, 0.03935557414283708, 0.021775078259213876, 0.037997478813511565, 0.03670308136440825, 0.0, 0.11722522630372034, 0.029753745208638938, 0.02704235332867677, 0.0, 0.015604780443500665, 0.13718113252042574, 0.0, 0.0223970898516459, 0.01824168402125048, 0.0, 0.0, 0.0, 0.04323960556514961, 0.03303757117705719, 0.01937886432831948, 0.003958783545891853]


# Filename: cmc.csv
# Score:  0.5057705363204344
# Info Gain:  [0.07090633894894594, 0.04013859922938412, 0.10173991727554088, 0.009820501434384843, 0.002582332379721608, 0.030474214560266555, 0.032511460053806784, 0.015786455595620197]


# Filename: car.csv
# Score:  0.8738425925925926
# Info Gain:  [0.09644896916961399, 0.07370394692148596, 0.004485716626632108, 0.2196629633399082, 0.030008141247605424, 0.26218435655426386]


# Filename: breast-cancer.csv
# Score:  0.7587412587412588
# Info Gain:  [0.010605956535614136, 0.0020016149737116518, 0.05717112532429669, 0.06899508808988597, 0.08012009687900967, 0.07700985251661441, 0.0024889884332655043, 0.015066622054149992, 0.025819023909141148]


# Filename: nursery.csv
# Score:  0.9026234567901235
# Info Gain:  [0.07293460750309988, 0.1964492804881155, 0.005572591715219843, 0.011886431475775838, 0.019602025022871672, 0.0043331270252002785, 0.022232616894018342, 0.9587749604699762]


# Filename: mushroom.csv
# Score:  0.9587641555883801
# Info Gain:  [0.04879670193537311, 0.028590232773772817, 0.03604928297620391, 0.19237948576121966, 0.9060749773839998, 0.014165027250616302, 0.10088318399657026, 0.23015437514804615, 0.41697752341613137, 0.007516772569664321, 0.4001378247172982, 0.2847255992184845, 0.2718944733927464, 0.2538451734622399, 0.24141556652756657, 0.0, 0.0238170161209168, 0.03845266924309054, 0.3180215107935376, 0.4807049176849154, 0.2019580190668524, 0.1568336046050921]


# %% [markdown]
# ## Number 1
#  Yes, it can. Information Gain is a way to measure how important an attribute
#  is to make predictions based on a set of data. The high value of an attribute's Information Gain means the attribute affects the class of an instance more. In the
#  given datasets, it can be seen that, if a classifier has more attributes with
#  relatively high Information Gain, the classifier tends to be more accurate.
#  However, there are outliers such as the hypothyroid.csv and
#  primary-tumor.csv.

#  In the hypothyroid.csv, Information Gain of each attribute is relatively low, and it should make the classifier less accurate. But, the accuracy is relatively high. This can be explained due to the high count of "negative" cases of hypothyroid in the dataset. This makes the classifier tend to predict "negative" which is true in most of the data. Due to this, Information Gain affects classifier prediction less.

#  In the primary-tumor.csv, the Information Gain of several attributes are relatively high, and with such values, the classifier should be able to make more accurate predictions. However, the accuracy is, in fact, the lowest. This is due to the high count of missing values in the dataset. Due to this, the value of Information Gain can be high as the missing values make the Mean Information smaller. This also makes it harder for the classifier to predict as the high count of missing values contributes to the problem of lack of data. Hence, the low accuracy.
# %%
# Holdout implementation for question number 4
def holdout(trainPercentage, filepath):
    files = prepFiles(filepath)

    # if asked for all data, return the same data as the test data and train
    # data
    if trainPercentage == 1:
        return files, files

    # getting the number of training data
    trainLength = int(len(files) * trainPercentage)

    # shuffling the files so that each instance is randomly assigned as either
    # training data or test data
    random.shuffle(files)

    return files[:trainLength], files[trainLength:]

# grab all the filepaths in the directory
files = collectFiles(
    "/Users/novan/Desktop/CODE/Machine Learning/assignment1/2019S1-proj1-data/"
    )

# train percentage to determine the percentage of training data
trainPercentage = 0.5

# split and evaluate data, then print the necessary data
for fp in files:
    trainData, testData = holdout(trainPercentage, fp)
    print("Filepath: ", fp)
    print("Score: ", float(evaluate(testData, trainData)))
    print("\n")

# %% [markdown]
# Using Holdout 80% as training data
# Filename: hypothyroid.csv
# Score:  0.957345971563981
# Higher by 0.6%

# Filename: primary-tumor.csv
# Score:  0.3088235294117647
# Lower by 13%

# Filename: hepatitis.csv
# Score:  0.8387096774193549
# Higher by 6%

# Filename: anneal.csv
# Score:  0.7722222222222223
# Lower by 4%

# Filename: cmc.csv
# Score:  0.46779661016949153
# Lower by 4%

# Filename: car.csv
# Score:  0.8323699421965318
# Lower by 4%

# Filename: breast-cancer.csv
# Score:  0.7931034482758621
# Higher by 4%

# Filename: nursery.csv
# Score:  0.9012345679012346
# Lower by 0.1%

# Filename: mushroom.csv
# Score:  0.9587692307692308
# Very close

# %% [markdown]
# Score with Holdout 50-50
# Filename: hypothyroid.csv
# Score:  0.9494310998735778
# Lower by 0.2%

# Filename: primary-tumor.csv
# Score:  0.2823529411764706
# Lower by 15%

# Filename: hepatitis.csv
# Score:  0.8333333333333334
# Higher by 0.1%

# Filename: anneal.csv
# Score:  0.8106904231625836
# Lower by 0.2%

# Filename: cmc.csv
# Score:  0.49525101763907736
# Lower by 1%

# Filename: car.csv
# Score:  0.8460648148148148
# Lower by 3%

# Filename: breast-cancer.csv
# Score:  0.7622377622377622
# Higher by 1%

# Filename: nursery.csv
# Score:  0.8976851851851851
# Lower by 1%

# Filename: mushroom.csv
# Score:  0.9455933037912359
# Lower by 1%
# %% [markdown]
# ## Number 4
# The holdout implementation takes in a percentage of data, in this case
#  80 % - 20 % and 50 % - 50 % of training data - test data. The data is shuffled
#  so that each instance is randomly assigned as either training data or
#  test data.

#  By splitting the data into training data and testing data, it is expected
# that the scores by implementing holdout evaluation would be lower than the test that uses training data as testing data. In this case, by comparing each score, it can be concluded that the expectation was correct. Most of the accuracy score tend to be lower than the score with the classifier that uses all data for training and testing. Some of the differences are small,
#  with only 0.1 % to 1 % of difference, but there are classifiers that implement
#  holdout with an even lower score, even as low as 15 % (the primary-tumor.csv 50-50). There are of course some cases where the classifier scores higher,
#  but the number of cases is smaller than the lower score cases. The difference is relatively insignificant. The most significant one would be the
#  80-20 hepatitis.csv classifier with the difference of 6 % with the classifier that uses all of the data as training data. This might be due to the lack of data which influence the probability calculations.
