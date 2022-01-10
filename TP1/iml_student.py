# coding=utf-8

#######################################################
# LOAD DATA
#######################################################

import os,glob
import numpy
import fitsio
from matplotlib import pyplot as plt

from sklearn import svm, metrics
import math, random

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


#######################################################
# VERIFICATION FUNCTIONS
#######################################################

#
# correctness
#
# Accuracy = Correctly classified / all
def accuracy(tp, fp, fn, tn):
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

# Precision = True Positives/ (True Positives + False Positives)
def precision(tp, fp, fn, tn):
    return tp / (tp + fp)


# Recall = True Positives/ (True Positives + False Negatives)
def recall(tp, fp, fn, tn):
    return tp / (tp + fn)

# F1 Score = (2 x Precision x Recall) / (Precision + Recall)
def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)

    return 2 * p * r / (p + r)
    
    
#####################################################
# END VERIFICATION FUNCTIONS
#####################################################



# Initialisation d'une liste pour contenir les images
# et d'une liste pour contenir les types morphologiques
raw_data =[]
type_morph =[]

 

# Définir les chemins vers les images et le catalogue,
mydataDir ='/home/TP/modules/sujets/IML/tp_galaxies/data'
mycatalog_path =os.path.join(mydataDir,'data-COSMOS-10000-id.txt')
mypath_template ='/home/TP/modules/sujets/IML/tp_galaxies/data/image/*'
 

# Chargement des données qui nous intéressent dans le catalogue
ids, mod = numpy.loadtxt(mycatalog_path, unpack=True, usecols=(0,3))

# Chargement des images
for one in glob.glob(mypath_template):
   # Extraction de l'id à partir du nom de fichier
   #print(one)
   idi =int(one.split('/')[-1].split('_')[0])
   #print(idi)
   modi = mod[ids==idi][0]
   #print(modi)
   # On va ignorer les out layer ie les mod == 0
   if modi>0:
      # Ajout de l'image
      data = fitsio.read(one)
      raw_data.append(data)
      # Ajout du type morphologique
      type_morph.append(modi)

# Reformatage en numpy pour plus de facilité
raw_data = numpy.asarray(raw_data)
type_morph = numpy.asarray(type_morph)

#######################################################
# display
#######################################################

# Un petit graphique pour illustrer
fig = plt.figure(figsize=(7,7))
for i in range(9):
   plt.subplot(330+i+1)
   plt.imshow(raw_data[i])
   plt.title('type %d'%(type_morph[i]))
   plt.axis('off')
# Visualiser le plot
plt.show()


#######################################################
# DATA TRANSFORMATION
#######################################################

# Normalisation des images
data_scaled = numpy.asarray([(img-img.mean())/img.std()for img in raw_data])

# Transformation en 1d array
data_1d = data_scaled.reshape((data_scaled.shape[0],-1))

# Vérifions avec la dimension des données sous forme de vecteurs
print(data_1d.shape)

# Et une valeur moyenne d'environ 0
print(data_1d.mean())


#######################################################
# DATA SPLIT
#######################################################

def split_data(data, prob):
    """split data into fractions [prob, 1 - prob]"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

# Returns n order data_train.image, data_test.image, data_train.result, data_test.result
# test_pct gives the percentage of data to use in test (1 = 100%))
def train_test_split(x, y, test_pct):
    data = zip(x, y)                              # pair corresponding values  
    train, test = split_data(data, 1 - test_pct)  # split the dataset of pairs
    x_train, y_train = zip(*train)                
    x_test, y_test = zip(*test)
    return x_train, x_test, y_train, y_test
	
# Use to equilibrate everything
def equilibring(data_images, data_results):
	temp = zip(data_images, data_results)
	#Create empty dictionnary
	counter_array={}
	result=[]
	nb_type_x = 0
	# Counting how many time every type appears
	for row in temp:
		# If key in dict
		if row[1] in counter_array:
			counter_array[row[1]] += 1
		else:
			#Create new key in dic using the type found number
			counter_array[row[1]] = 1
			
	#Printing the dict of things found
	print(counter_array.items())
	
	# Finding the one result with the minimal in the values of the dict
	minimumkey = min(counter_array.keys(), key=(lambda k: counter_array[k]))
	print("Minimal value found is ", counter_array[minimumkey], " (", minimumkey, ")")
	
	# Initialize every counter of removal
	minimal_counter = {}
	for key in counter_array:
		minimal_counter[key] = counter_array[key] - counter_array[minimumkey]
		if minimal_counter[key] == minimumkey:
			minimal_counter[key] = 0
	
	#Print how many to remove of each
	print("Removing: ")
	print(minimal_counter.items())
	
	# Cloning to a new array
	temp = zip(data_images, data_results)
	for row in temp:
		if minimal_counter.get(row[1]) == 0:
			result.append(row)
		else:
			minimal_counter[row[1]] -= 1
			
	
	#Replacing it to to regular arrays
	data_images, data_results = zip(*result)
	return data_images, data_results
		

#######################################################
# WORK
#######################################################

# Nombre de donnees
n_samples = len(data_1d)

# equilibring types sample
data_1d, type_morph = equilibring(data_1d, type_morph)

# Split data in train and test arrays.
#train_test_split([Images Entre], [Resultats associes Entre], [% a utiliser en test)
data_train_image, data_test_image, data_train_result, data_test_result = train_test_split(data_1d, type_morph, 0.3)


###########################################
#	TRAINING PART
###########################################

parameter = 1
parameter_max_value = 10
parameter_step = 1

best_score = 0
best_parameter = 0


while parameter < parameter_max_value:
	# Create a classifier: a support vector classifier, first as a black box
	#clf = svm.SVC(gamma=0.001, C=100.)
	clf = RandomForestClassifier(max_depth=parameter, random_state=0)
	#clf = AdaBoostClassifier(n_estimators=100, random_state=0)
	#clf = GradientBoostingClassifier(random_state=0)
	#MLPClassifier. Default hidden layers nb is 100. Each separator is the layer, (100, ) means 100 hidden neurons on the first layer, and no 2nd layer
	#clf = MLPClassifier(hidden_layer_sizes=(1000, 1000, 1000))

	#Separation de l'etude en 2 groupe, associant les images resolues et leur resolution
	clf.fit(data_train_image, data_train_result)

	# Now predict the value of the digit on the second half:
	expected = data_test_result
	predicted = clf.predict(data_test_image)
	
	#Check if new parameter is better
	results = metrics.classification_report(expected, predicted, output_dict=1)
	result_accuracy = results.get("accuracy")
	if result_accuracy > best_score:
		best_score = result_accuracy
		best_parameter = parameter
	print("parameter : ", parameter, "   Acc : ", result_accuracy, "              Best accuracy : ", best_score) 
	parameter = parameter + parameter_step


###########################################
#	FINAL CLEAN PRINT
###########################################
print("Best parameter found: ", best_parameter) 

clf = RandomForestClassifier(max_depth=best_parameter, random_state=0)

#Separation de l'etude en 2 groupe, associant les images resolues et leur resolution
clf.fit(data_train_image, data_train_result)

# Now predict the value of the digit on the second half:
expected = data_test_result
predicted = clf.predict(data_test_image)
print("Classification report for classifier %s:\n%s\n" \
   % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

#######################################################
# SVM
#######################################################
"""
# Results
#              precision    recall  f1-score   support

#         1.0       0.96      0.66      0.78       249
#         2.0       0.74      0.99      0.85       763
#         3.0       0.00      0.00      0.00       183

#    accuracy                           0.77      1195
#   macro avg       0.57      0.55      0.54      1195
#weighted avg       0.67      0.77      0.70      1195


#Confusion matrix:
#[[164  85   0]
# [  6 757   0]
# [  0 183   0]]
"""

#SVM mauvais, pad adapté au probleme car trop de type 2. Peut etre pas adapte au probleme Mieux après équilibrage mais pas extra


#######################################################
# RANDOMFOREST
#######################################################
"""
# Results
#              precision    recall  f1-score   support

#         1.0       0.92      0.82      0.87       210
#         2.0       0.63      0.56      0.59       212
#         3.0       0.63      0.78      0.70       195

#    accuracy                           0.72       617
#   macro avg       0.73      0.72      0.72       617
#weighted avg       0.73      0.72      0.72       617


#Confusion matrix:
#[[172  27  11]
# [ 14 119  79]
# [  0  43 152]]
"""

#Meilleur resultat

#######################################################
# AdaBoostClassifier
#######################################################
"""
# Results
#              precision    recall  f1-score   support

#         1.0       0.86      0.40      0.54       201
#         2.0       0.46      0.74      0.56       203
#         3.0       0.73      0.66      0.69       199

#    accuracy                           0.60       603
#   macro avg       0.68      0.60      0.60       603
#weighted avg       0.68      0.60      0.60       603


#Confusion matrix:
#[[ 80 113   8]
# [ 11 150  42]
# [  2  65 132]]
"""






