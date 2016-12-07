
### Fraud at Enron Using Emails and Financial Data

## Sergei Neviadomski

## Intro
Enron Corporation was an American energy, commodities, and services company
based in Houston, Texas. It was founded in 1985. Before its bankruptcy on 
December 2, 2001, Enron employed approximately 20,000 staff and was one of the
world's major electricity, natural gas, communications and pulp and paper 
companies, with claimed revenues of nearly $101 billion during 2000. Fortune 
named Enron "America's Most Innovative Company" for six consecutive years.

At the end of 2001, it was revealed that its reported financial condition was 
sustained by institutionalized, systematic, and creatively planned accounting 
fraud, known since as the Enron scandal. Enron has since become a well-known 
example of willful corporate fraud and corruption. The scandal also brought into
question the accounting practices and activities of many corporations in the 
United States and was a factor in the enactment of the Sarbanes-Oxley Act of
2002. The scandal also affected the greater business world by causing the 
dissolution of the Arthur Andersen accounting firm.

## Removing outliers
In this project, I use machine learning to identify persons of interest based 
on financial and email data made public as a result of the Enron scandal. As I 
said we have financial data and emails of top enron's employees. Here is some 
information about dataset:
Total number of features: 20
Total number of data points: 146
Number of POI/Non-POI data points: 18/128
We have a lot of missing values. Features with highest persentage of missing 
value are 'loan_advances', 'director_fees' and 'restricted_stock_deferred'.
As a first step I checked all this data for outliers and missing values. After 
this step I removed 3 rows of data: 'THE TRAVEL AGENCY IN THE PARK', 'TOTAL', 
'LOCKHART EUGENE E'. First two rows are not personal data. Third one contains 
only missing values. 

## Adding features 
After scrutinizing my data I decided to add 3 features. First one is the first 
component of PCA of all financial features. I did this since we have many 
financial features and all of them are highly correlated. Using all of this 
features in our models could lead to bad performance. In my opinion, the first 
component of PCA will catch all the best from all the financial features.
   Then I added percentage of outcoming emails to POI and percentage of 
incoming emails from POI. Relative values of emails to/from POI is better
choice than absolute values. 
   After that I ran SelectKBest algorithm that showed me scores of features.
Here is top of features:
salary 18.3
bonus 20.8
total_stock_value 24.2
exercised_stock_options 24.8
PCA1 14.5
from_POI_percent 115.0
to_POI_percent 0.15
I prefer not to rely on algorithm when choosing features, but to take results 
of algorithms into consideration when choosing features for my prediction model 
that intuitively fit this model. First four features look related to each other.
So I decided to use only two of them: salary and exercised_stock_options. I want
to use them because salary is conventional measure to assess compensation and
exercised_stock_options is good choice to measure ability to use internal
information for private gains. Both of them have high score. I also want to use
PCA1 as aggregation of all financial features. From_POI_percent and 
to_POI_percent is perfect to measure extent of communication between POIs. So
this five features is my first pick. I tried random forest algorithm on them and
get pretty good results: Accuracy:0.96, Precision:0.92, Recall:0.74. Using top 5
features from SelectKBest algorithm or adding more features improves results by 
less then 1 percent. That's why I prefer to keep my sensible pick of features. 
On the other side removing features lead to significant reducing of scores, 
especially recall by 0.2 and more. So I'll use my pick for further testing on 
different algorithms.  
Final features: 'salary', 'exercised_stock_options', 'PCA1', 'from_POI_percent',
'to_POI_percent'. 

## Choosing right algorithm
I tried 8 different classification algorithms. Quadratic discriminant analysis 
showed worst result: Accuracy:0.77553, Precision:0.14931, Recall:0.14550, 
F1:0.14738, F2:0.14625. Random forest and Desision tree showed best results.
Desision tree: Accuracy:0.94547, Precision:0.80246, Recall:0.78400, F1:0.79312, 
F2: 0.78762; Random forest: Accuracy:0.95427, Precision:0.93167, Recall:0.70900, 
F1:0.80522, F2:0.74459. This two results are pretty close, so I decided to 
choose both of them for further tuning.

## Tuning
Tuning is a process of choosing right parameters for machine learning algorithm. 
There are two possible problems. Depending on parameters model can be overfitted 
which mean that it will try to catch every point in training data but not the 
general trend. On the other side model could be too simple, so it wouldn't 
predict values on testing data.
In my case I've chosen 2 algorithms for tuning. I used GridSearchCV module from 
sklearn library for tuning and comparison results. As I predicted tuning of 
Random Forest didn't improve model significantly. Decision  Trees is more prone 
to tuning so I got much better results from tuning of this model. But in spite 
of this fact Random Forest showed slightly better results.

## Choosing final model
Validation is performed to ensure that a machine learning algorithm generalizes 
well. There're two main types of validation processes. We could split our data 
into two sets: training set and testing set, train our data on first one and 
then test final model on testing set. But I used second type. That's k-fold 
cross-validation. In this type of validation we devide data into K parts and 
each part apears as testing set. Then we calculate average result from k tests. 
Choosing right metrics is extremely important for validation of any model. In my 
case I used 5 different metrics: accuracy, precision, recall, F1 and F2. But
most attention I paid to precision and recall. In my project precision is 
fraction of predicted POIs that are actual POIs. And recall is fraction of 
actual POIs that are predicted. We can always improve one of them by sacrificing 
another, while best model is model that maximizes both of them. My best model is 
random forest model, that has precision of 0.93 and recall of 0.72.