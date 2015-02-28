#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

cdict = {'red': ((0., 1, 1),
                 (1, 0.5, 0.5)),
         'green': ((0., 1, 1),
                   (1, 0, 0)),
         'blue': ((0., 1, 1),
                  (1, 0, 0))}

my_cmap = mcolors.LinearSegmentedColormap('my_colormap',cdict,256)

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'fraction_from_poi',
                 'exercised_stock_options',
                 'shared_receipt_with_poi',
                 'expenses',
                 'other']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop( 'TOTAL', 0 )
### Task 3: Create new feature(s)
###Create 3 new features 
#get the top 3 bonuses and top 3 salaries and put into set
from sets import Set
specific_emails = Set() 
tmp = []
for key in data_dict.keys():
    if data_dict[key]["salary"] != 'NaN' and data_dict[key]["bonus"] != 'NaN' and data_dict[key]["poi"] == True:
        tmp.append({"key": key, "salary": data_dict[key]["salary"],"bonus":data_dict[key]["bonus"],"email_address":data_dict[key]["email_address"]})

###Getting emails of n top of persons sortered by feature_name feature
###Receiving Set of emails something like (['tim.belden@enron.com', 'kenneth.lay@enron.com', 'jeff.skilling@enron.com'])
def add_to_specific_email_set(feature_name,n):
    cleaned_data = sorted(tmp, key=lambda (x): x[feature_name], reverse = True)
    cleaned_data  = cleaned_data[0:n]
    for email in cleaned_data:
        specific_emails.add(email["email_address"])    
    
add_to_specific_email_set("salary",5)
add_to_specific_email_set("bonus",5)

###getting all to_emails from current message
to_emails_dict = {}
def getToEmails(fname):
    with open(fname) as f:
        content = f.readlines()
        for c in content:
            if 'To:' in c:
                to_emails = c.replace(" ", "").split(':')[1][:-2].split(',')
                break
    return to_emails

###counting all messages from specific emails
for email in specific_emails:
    try:
        list_messages = open(os.getcwd()+"/emails_by_address/"+"from_"+email+".txt", "r")
        for path_str in list_messages:
            path = os.path.join('..', path_str[:-1])
            to_addresses = getToEmails(path)
            for addr in to_addresses:
                if addr not in to_emails_dict.keys():
                    to_emails_dict[addr] = {"from_specific_email":1}
                else:
                    to_emails_dict[addr]["from_specific_email"] += 1
    except:
        pass

###compute fraction 
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    if poi_messages != 'NaN' and all_messages != 'NaN' and all_messages != 0:
        fraction = float(poi_messages)/float(all_messages)
    else:
        fraction = 0.
    return fraction

###creating 3 new features fraction_from_poi, fraction_to_poi (like in example) and from_specific_email
for key in data_dict:
    data_point = data_dict[key]
    email_key = data_point["email_address"]
    data_point["fraction_from_poi"] = computeFraction( data_point["from_this_person_to_poi"], data_point["from_messages"] )
    data_point["fraction_to_poi"] = computeFraction( data_point["from_poi_to_this_person"], data_point["to_messages"] )
    if email_key in to_emails_dict.keys():
        data_point["from_specific_email"] = to_emails_dict[email_key]["from_specific_email"]
    else:
        data_point["from_specific_email"] = 0

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
"""
from_poi = []
to_poi = []
color = []
for point in data:
    from_poi.append(point[1])
    to_poi.append(point[2])
    color.append(point[0])
plt.scatter( from_poi, to_poi, c=color, cmap=my_cmap )
plt.xlabel("fraction_to_poi")
plt.ylabel("fraction_from_poi")
plt.show()
"""

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

"""Uncomment to try 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features, labels)
"""
"""Uncomment to try 
from sklearn.svm import SVC
clf = SVC(C=1.0, kernel="rbf")
clf.fit(features, labels)
"""

from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=42, min_samples_split=12,max_depth=6, splitter='best')
clf = clf.fit(features, labels)

###feature importances for feature selection process
feature_importances = clf.feature_importances_
sorted_idx = (-np.array(feature_importances)).argsort()
print "Rank of features"
for idx in sorted_idx:
    print "{:4f} : {}".format(feature_importances[idx], features_list[idx+1])


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
"""
from sklearn import grid_search
parameters = {'min_samples_split':[2,4,6,8,10,12,50],
              'splitter': ('best','random'),
              'max_depth':[None,2,4,6,8,10,15,20]
              }
clf_s = grid_search.GridSearchCV(tree.DecisionTreeClassifier(), parameters).fit(features, labels)
print 'best estimator:'
print clf_s.best_estimator_
"""

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
