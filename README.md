#Udacity Nanodegree. Project 4. Identifying Fraud from Enron Email
##1. Dataset and goal of project
####Goal
The main purpose of project is develop the machine learning algorithm to detect person of interest(POI) from dataset.
A POI is someone who was indicted for fraud, settled with the government, or testified in exchange for immunity.

####Dataset
We have Enron email+financial (E+F) dataset. It contains 146 Enron managers to investigate. Each sample in this dictionary containing 21 features. 
18 people from this dataset labeled as POI. All of them have `poi` feature set as **1**. There's two imbalanced classes (many more non-POIs than POIs).

There's example of one POI data point: 


####Outliers
Dataset contains some outliers. The TOTAL row is the biggest Enron E+F dataset outlier. We should remove it from dataset for reason it's a spreadsheet quirk.
Moreover, there’s 4 more outliers with big salary and bonus. Two people made bonuses more than 6 million dollars, and a salary of over 1 million dollars. 
There's no mistake. Ken Lay and Jeffrey Skilling made such money. So, leave these data points in and examine it with others.

##2. Feature selection process
####I selected the following features
`exercised_stock_options`  `shared_receipt_with_poi`  `fraction_from_poi`  `expenses`  `other`

####New features
In addition I create two new features which were considered in course:
* `fraction_from_poi` fraction of messages to that person from a POI
* `fraction_to_poi` fraction of messages from that person to a POI
They created on assumption POI have more intensive correspondence to each other. 

Also I create another two features. Messages to/from current person from/to specific email addresses, which belong to four outliers (e.g. Ken Lay etc.)

* `from_specific_email` 
* `to_specific_email` 

Feature selection process include several iterations. 
On the first step I created set of features based on data visualization and intuition. Then I examine three classificator on this features. Dtecision Trees was selected as main algorithm. 
Since I choose Decision Trees as a classificator, I used feature importance method to optimize features for this dataset. 

As a result I’ve received the following feature importances:

    0.199176954733 1 exercised_stock_options
    0.179079931233 2 shared_receipt_with_poi
    0.133721510439 3 fraction_from_poi
    0.226273882385 4 expenses
    0.261747721211 5 other

fraction_from_poi looks like important feature and I leave it in. fraction_to_poi has importance equal to zero, so I've took in out from algorithm. Pearson who received e-mails from POI more often than others looks like POI himself. I also tried to change some features by hand but each time received worse performance.

To see full log of feature selection process go to the next block of this page.

##3. Pick an algorithm
I tried the Naive Bayes, SVM and Decision Trees algorithms. 

####All results of examination I included in the following table

|Algorithm|Accuracy|Precisions|Recall|
|:---|---|---|---|
|**Naive Bayes**|0.32913|0.15392|0.89650|
|**Decision Trees**|0.80093|0.25151|0.24950|
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

As a result, I received better performance with `min_samples_split` = '2' and `Splitter` = 'best' `and max_depth` = 

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
|Precision|0.45046|
|Recall|0.42050|
|Accuracy |0.84393|
|True positives |810|
|False positives|678|
|False negatives|1190|
|True negatives|11322|
	
####Conclusion
Precision and Recall have almost identical values and both higher than .3. Thus, project goal was reached.
Precision 0.51 means when model detect person as POI it was true only in 51% cases. 
At the same time Recall = 0.48 says only 48% of all POIs was detected.

We have very imbalanced classes in E+F dataset. In addition, almost half of all POIs weren't included in dataset. 
In such conditions result we received good enough, but it's not perfect, of course.

#Feature selection log

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
