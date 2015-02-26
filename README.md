#Udacity Nanodegree. Project 4. Identifying Fraud from Enron Email
##Dataset and goal of project
###Goal
The main purpose of project is develop the machine learning algorithm to detect person of interest(POI) from dataset.
A POI is someone who was indicted for fraud, settled with the government, or testified in exchange for immunity.

###Dataset
We have Enron email+financial (E+F) dataset. It contains 146 Enron managers to investigate. Each value in this dictionary containing all the features of person. There are 21 features. 
18 people labeled as POI. All of them have `poi` feature settled as **1**. As we can see there're imbalanced classes (many more non-POIs than POIs).

There's example of one POI data point: 


###Outliers
Dataset contains some ouliers. The TOTAL row is the biggest Enron E+F dataset outlier. We should remove it from dataset for reason it's a spreadsheet quirk.
Moreover, there’s 4 more outliers with big salary and bonus. Two people made bonuses more than 6 million dollars, and a salary of over 1 million dollars. 
There's no mistake. Ken Lay and Jeffrey Skilling made such money. So, leave these data points in and examine it with others.

##Feature selection process
###I selected the following features:
`exercised_stock_options`, `shared_receipt_with_poi`, `fraction_from_poi`, `expenses`, `other`

Feature selection process include about 10 iterations. I choose features based on visualization of data, Decisions Tree feature importances and simply by hand.
In Intro to Machine Learning course we figured out using only financial features may leads to wrong results. Thus, I examine financial and emails features together. 

###New features
In addition I create two new features:
* `fraction_from_poi` fraction of messages to that person that are from a POI
* `fraction_to_poi` fraction of messages from that person that are to a POI

They created on assumption POI have more intensive correspondence to each other. 

###Final feature selection algorithm
Since I choose Decisions Trees as a final algorithm, I used feature importance method to select best features for this dataset. 
I optimized 

fraction_from_poi looks like important feature and I leave it in. fraction_to_poi has importance equal to zero, so I've took in out from algorithm.
Pearson who received e-mails from POI more often than others looks like POI himself.

I’ve received the following feature importances:

    0.199176954733 1 exercised_stock_options
    0.179079931233 2 shared_receipt_with_poi
    0.133721510439 3 fraction_from_poi
    0.226273882385 4 expenses
    0.261747721211 5 other

I also tried to change some features by hand but each time received worse performance.
To see full log of feature selection process go to the next block of this page.

##Pick an algorithm
I tried the Naive Bayes, SVM and Decisions tree algorithms. 

###All results I include in the following table
|Algorithm	|accuracy	|precisions|	Recall |
|:---|---|---|---|
|**Naive Bayes**|0.32913|0.15392|0.89650|
|**Decisions Tree**|0.80093|0.25151|0.24950|
|**SVM**|-|-|-|

SVM algorithm returned the following error :
```
Got a divide by zero when trying out: SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
```

###Choosen algorithm
Based on performance level I picked Decisions Tree as a final algorithm.

##Tune the algorithm
###Reasons for algorithm tuning
The main reason is get better results from algorithm. Parameters of ML classifiers have a big influence in output results. 
The purpose of tuning is to find best sets of parameters for particular dataset.

###GridSearchCV
I apply GridSearchCV to tune the following parameters

|Parameter          |Settings for investigation |
|:------------------|:--------------------------|
|min_samples_split	| [2,6,8,10]                | 
|Splitter	        | (random,best)             |
|max_depth	        | [None,2,4,6,8,10,15,20]   |

As a result, I received better performance with `min_samples_split` = '2' and `Splitter` = 'best'

##Validation
According to [sklearn documentation][sklearn_mistake] one of the main and classical mistakes in validation is using the same data for training and testing. 
>Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: 
>a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting.

To validate my analysis I applied [stratified shuffle split cross validation][StratifiedShuffleSplit] developed by Udacity and defined in tester.py file

##Evaluation metrics
I used precision and recall evaluation metrics to estimate model.
Decisions Tree algorithm with choosen features and tuned parameters has

    Precision = 0.45046     Recall = 0.42050    Accuracy = 0.84393
	True positives:  810	False positives:  678	False negatives: 1190	True negatives: 11322
###Conclusion
Precision and Recall have almost identical values. In other words we can say.
Мы сгенерировали 14000 экспериментов. Из них правильно были идентифицированы 810 POI, в тоже самое время ошибочно POI были названы 678 человек. 
В тоже самое время, наша модель не смогла определить POI 1190 раз

Как я отметил вначале, наши данные разбалансированы. К тому же не все POI включены в dataset. Это создает определенные трудности для качественного анализа. 
Мы должны стремится к повышению true/positive, т.к. основная задача - распознавание POIs.

We should make true positive rate higher to good flagging POI when it's present in the test data.


##Algorithm outputs log

##Related links
- [Documentation of scikit-learn 0.15][1]
- [sklearn tutorial][2]
- [Recursive Feature Elimination][3]
- [Selecting good features – Part I: univariate selection][4]
- [Accuracy, Precision and Recall(in Russian)][5] 

[1]: http://scikit-learn.org/stable/documentation.html
[2]: http://amueller.github.io/sklearn_tutorial/
[3]: http://topepo.github.io/caret/rfe.html
[4]: http://blog.datadive.net/selecting-good-features-part-i-univariate-selection/
[5]: http://bazhenov.me/blog/2012/07/21/classification-performance-evaluation.html
[StratifiedShuffleSplit]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
[sklearn_mistake]: http://scikit-learn.org/stable/modules/cross_validation.html 
