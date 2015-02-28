#Udacity Nanodegree. Project 4. Identifying Fraud from Enron Email
##1. Dataset and goal of project
####Goal
The main purpose of project is develop the machine learning algorithm to detect person of interest(POI) from dataset.
A POI is someone who was indicted for fraud, settled with the government, or testified in exchange for immunity.

####Dataset
We have Enron email+financial (E+F) dataset. It contains 146 Enron managers to investigate. Each sample in this dictionary containing 21 features. 
18 people from this dataset labeled as POI. All of them have `poi` feature set as **True**. There's two imbalanced classes (many more non-POIs than POIs).

There's example of one POI data point: 

	[{"SKILLING JEFFREY K":{
				'salary': 1111258, 
				'to_messages': 3627, 
				'deferral_payments': 'NaN', 
				'total_payments': 8682716, 
				'exercised_stock_options': 19250000, 
				'bonus': 5600000, 
				'restricted_stock': 6843672, 
				'shared_receipt_with_poi': 2042, 
				'restricted_stock_deferred': 'NaN', 
				'total_stock_value': 26093672, 
				'expenses': 29336, 
				'loan_advances': 'NaN', 
				'from_messages': 108, 
				'other': 22122, 
				'from_this_person_to_poi': 30, 
				'poi': True, 
				'director_fees': 'NaN', 
				'deferred_income': 'NaN', 
				'long_term_incentive': 1920000, 
				'email_address': 'jeff.skilling@enron.com', 
				'from_poi_to_this_person': 88
				}
	}]

####Outliers
Dataset contains some outliers. The TOTAL row is the biggest Enron E+F dataset outlier. We should remove it from dataset for reason it's a spreadsheet quirk.
Moreover, there’s 4 more outliers with big salary and bonus. Two people made bonuses more than 6 million dollars, and a salary of over 1 million dollars. 
There's no mistake. Ken Lay and Jeffrey Skilling made such money. So, leave these data points in and examine it with others.

##2. Feature selection process
####I selected the following features
`exercised_stock_options`  `shared_receipt_with_poi`  `fraction_from_poi`  `expenses`  `other` `salary`

####New features
In addition I create two new features which were considered in course:
* `fraction_from_poi` fraction of messages to that person from a POI
* `fraction_to_poi` fraction of messages from that person to a POI
They created on assumption POI have more intensive correspondence to each other. 

Also I create another feature. Messages to current person from specific email addresses, which belong to four POI outliers (e.g. Ken Lay etc.)
* `from_specific_email` 

Feature selection process include several iterations. 
On the first step I created set of features based on data visualization and intuition. Then I examine three classificator on this features. Dtecision Trees was selected as main algorithm. 
Since I choose Decision Trees as a classificator, I used feature importance method to optimize features for this dataset. 

As a result I’ve received the following feature importances:

	Rank of features
	0.224388 : other
	0.217197 : exercised_stock_options
	0.195282 : shared_receipt_with_poi
	0.185831 : expenses
	0.145820 : fraction_from_poi
	0.031483 : salary

fraction_from_poi looks like important feature and I leave it in. fraction_to_poi has importance equal to zero, so I've took in out from algorithm. Pearson who received e-mails from POI more often than others looks like POI himself. I also tried to change some features by hand but each time received worse performance.

To see full log of feature selection process go to the next block of this page.

##3. Pick an algorithm
I tried the Naive Bayes, SVM and Decision Trees algorithms. 

####All results of examination I included in the following table

|Algorithm|Accuracy|Precisions|Recall|
|:---|---|---|---|
|**Naive Bayes**|0.82100|0.29026|0.23700|
|**Decision Trees**|0.82620|0.34617|0.34150|
|**SVM**|-|-|-|

SVM algorithm returned the next error :
```
Got a divide by zero when trying out: SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
```

####Chosen algorithm
Based on best performance level I picked Decision Trees as a final algorithm.

##4. Tune the algorithm
####Reasons for algorithm tuning
The main reason is get better results from algorithm. Parameters of ML classifiers have a big influence in output results. 
The purpose of tuning is to find best sets of parameters for particular dataset.

####GridSearchCV
I apply GridSearchCV to tune the following parameters

|Parameter          |Settings for investigation |
|:------------------|:--------------------------|
|min_samples_split	| [2,6,8,10]                | 
|Splitter	        | (random,best)             |
|max_depth	        | [None,2,4,6,8,10,15,20]   |

As a result, I received better performance with `min_samples_split` = '12' and `Splitter` = 'best' `and max_depth` = '6'

##5. Validation
According to [sklearn documentation][sklearn_mistake] one of the main and classical mistakes in validation is using the same data for both training and testing. 
>Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: 
>a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting.

To validate my analysis I used [stratified shuffle split cross validation][StratifiedShuffleSplit] developed by Udacity and defined in tester.py file

##6. Evaluation metrics
I used precision and recall evaluation metrics to estimate model.
Final results can be found in table below

|Metric|Value|
|:----|:----|
|**Precision**|**0.52166**|
|**Recall**|**0.41550**|
|Accuracy |0.86207|
|True positives |831|
|False positives|762|
|False negatives|1169|
|True negatives|11238|
	
####Conclusion
Precision and Recall have almost identical values and both higher than .3. Thus, project goal was reached.
Precision 0.52166 means when model detect person as POI it was true only in 52% cases. 
At the same time Recall = 0.41550 says only 41% of all POIs was detected.

We have very imbalanced classes in E+F dataset. In addition, almost half of all POIs weren't included in dataset. 
In such conditions result we received good enough, but it's not perfect, of course.

#Algorithm outputs log
####STEP1. Init stage. Select classificator
#####features_list: 
	features_list = ['poi',
                 'fraction_to_poi',
                 'fraction_from_poi',
                 'from_specific_email',
                 'from_messages',
                 'exercised_stock_options',
                 'shared_receipt_with_poi',
                 'expenses',
                 'other',
                 'bonus',
                 'salary',
                 'total_stock_value'] 
				 
#####Classificator:
	clf = tree.DecisionTreeClassifier()
#####Metrics:
	Accuracy: 0.82620	Precision: 0.34617	Recall: 0.34150	F1: 0.34382	F2: 0.34242
	Total predictions: 15000	
	True positives:  683	False positives: 1290	
	False negatives: 1317	True negatives: 11710
#####Classificator:
	clf = GaussianNB()
#####Metrics:
	Accuracy: 0.82100	Precision: 0.29026	Recall: 0.23700	F1: 0.26094	F2: 0.24603
	Total predictions: 15000	
	True positives:  474	False positives: 1159	
	False negatives: 1526	True negatives: 11841

####STEP2. Select features by Decision Trees feature_importances_
#####feature_importances_
	Rank of features
	0.260361 : other
	0.231356 : exercised_stock_options
	0.225797 : expenses
	0.134677 : fraction_from_poi
	0.118998 : shared_receipt_with_poi
	0.028810 : salary
	0.000000 : fraction_to_poi
	0.000000 : from_specific_email
	0.000000 : from_messages
	0.000000 : bonus
	0.000000 : total_stock_value

#####Metrics after optimizing
	Accuracy: 0.84029	Precision: 0.43873	Recall: 0.42250	F1: 0.43046	F2: 0.42565
	Total predictions: 14000	
	True positives:  845	False positives: 1081	
	False negatives: 1155	True negatives: 10919

####STEP3. Tune the algorithm
#####features_list
	features_list = ['poi',
                 'salary',
                 'fraction_from_poi',
                 'exercised_stock_options',
                 'shared_receipt_with_poi',
                 'expenses',
                 'other'] 
#####best estimator:
	DecisionTreeClassifier(compute_importances=None, criterion='gini',
            max_depth=6, max_features=None, max_leaf_nodes=None,
            min_density=None, min_samples_leaf=1, min_samples_split=12,
            random_state=None, splitter='best')
#####Metrics after tuning (BEST CHOISE!)
	Accuracy: 0.86207	Precision: 0.52166 Recall: 0.41550 F1: 0.46257	F2: 0.43313
	Total predictions: 14000	
	True positives:  831	False positives:  762	
	False negatives: 1169	True negatives: 11238

####STEP4. Change features by hand (examine only email features)
	features_list = ['poi',
                 'fraction_from_poi',
                 'fraction_to_poi',                 
                 'exercised_stock_options',
                 'shared_receipt_with_poi'] 
#####Metrics 
	Accuracy: 0.83108	Precision: 0.43510	Recall: 0.32850	F1: 0.37436	F2: 0.34543
	Total predictions: 13000	
	True positives:  657	False positives:  853	
	False negatives: 1343	True negatives: 10147
	
####STEP5. Tune parameters by hand
#####parameters
	clf = tree.DecisionTreeClassifier(random_state=42, min_samples_split=2,max_depth=2, splitter='best')
#####Metrics 
	Accuracy: 0.83550	Precision: 0.37947	Recall: 0.23850	F1: 0.29291	F2: 0.25764
	Total predictions: 14000	
	True positives:  477	False positives:  780	
	False negatives: 1523	True negatives: 11220
	
####STEP6. Final choise
#####features_list
	features_list = ['poi',
                 'fraction_from_poi',
                 'fraction_to_poi',                 
                 'exercised_stock_options',
                 'shared_receipt_with_poi'] 
#####parameters				 
	clf = tree.DecisionTreeClassifier(random_state=42, min_samples_split=12,max_depth=6, splitter='best'	
#####Metrics 	
	Accuracy: 0.86207	Precision: 0.52166	Recall: 0.41550	F1: 0.46257	F2: 0.43313
	Total predictions: 14000	
	True positives:  831	False positives:  762	
	False negatives: 1169	True negatives: 11238
	
#Related links
- [Documentation of scikit-learn 0.15][1]
- [sklearn tutorial][2]
- [Recursive Feature Elimination][3]
- [Selecting good features – Part I: univariate selection][4]
- [Cross-validation: the right and the wrong way][6]
- [Accuracy, Precision and Recall(in Russian)][6] 

[1]: http://scikit-learn.org/stable/documentation.html
[2]: http://amueller.github.io/sklearn_tutorial/
[3]: http://topepo.github.io/caret/rfe.html
[4]: http://blog.datadive.net/selecting-good-features-part-i-univariate-selection/
[5]: https://www.kaggle.com/c/the-analytics-edge-mit-15-071x/forums/t/7837/cross-validation-the-right-and-the-wrong-way
[6]: http://bazhenov.me/blog/2012/07/21/classification-performance-evaluation.html
[StratifiedShuffleSplit]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
[sklearn_mistake]: http://scikit-learn.org/stable/modules/cross_validation.html 