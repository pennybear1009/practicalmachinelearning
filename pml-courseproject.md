# Practical Machine Learning Course Project
Xiaoning PEI  

##Executive Summary
People use devices such as Jawbone Up, Nike FuelBand, and Fitbit, to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this report, our goal is to use data collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, build a model to predict if participants performed correctly or incorrectly(Class A corresponds to the specified execution of the exercise, while the other 4 classes B-E correspond to common mistakes).

Read more: [http://groupware.les.inf.puc-rio.br/har#dataset#ixzz3vDjtjhmW]

##Data Preprocessing  
Before building models, we did the following data preprocessing on training and testing set:

+ **Loading data**: we loaded "pml-training.txt" as training and "pml-testing" as testing.

+ **Coercing other types to numeric**: there are some features with incorrect type in both training and testing set, so we need to transform these factor or logical vectors into numeric for better prediction performance. we choose to coerc 8-159 columns to numeric.

+ **Imputing NA variables**: Firstly, we discarded the predictors with NA involved in training set, and did the same on teseting set. Secondly we check if there is any near zero variables and remove them from both training and testing set. After doing this, there are only 58 predictors left.

+ **Slicing data**: We seperated training data into two sets with 60% for train and 40% for test.


```r
# loading data
library(caret)
training <- read.csv("G:/Coursera-DataScienceSpecialization/8-PracticalMachineLearning/CourseProject/pml-training.csv")
testing <- read.csv("G:/Coursera-DataScienceSpecialization/8-PracticalMachineLearning/CourseProject/pml-testing.csv")
# coerce other types to numeric
for (i in 8:159) training[,i] <- as.numeric(training[,i]) #left classe and user_name as factor
for (i in 8:159) testing[,i] <- as.numeric(testing[,i])
# imputing NA variables
# first step: remove all NA predictors and 93 left and remove the index column
testing <- testing[,-c(1,which(c(apply(training,2,anyNA))))]
training <- training[,-c(1,which(c(apply(training,2,anyNA))))]
# second step: remove near zero varialbes. 59 variables left for model building
nzv <- nearZeroVar(training,saveMetrics = TRUE) 
training <- training[,which(nzv$nzv==FALSE)]
testing <- testing[,which(nzv$nzv==FALSE)]
# Data Slicing: train-test(60-40)
Train <- createDataPartition(training$classe,p=.6,list = FALSE)
train <- training[Train,]
test <- training[-Train,]
```

## Model building and selection

+ **Training on train set**: we choose **Stochastic Gradient Boosting**(gbm) and **Random Forest**(rf) to train the data, and used 3-fold cross validation to perform resampling. we get two models here: fit.gbm and fit.rf


```r
fit.gbm <- train(classe~.,method="gbm",data=train,trControl=trainControl(method = "cv",number=3))
fit.rf <- train(classe~.,method="rf", data=train, trControl=trainControl(method = "cv",number=3))
```

+ **Predicting on test set**: we use the two models built in the former step to predict on test set, and get two results: p.gbm and p.rf 


```r
p.gbm <- predict(fit.gbm,test)
p.rf <- predict(fit.rf,test)
```

+ **Checking the accuracy**: we compared both out of sample accuracy, and Random Forest performs better than GBM. Here is the summary of the results of models.

1.Gradient Boosting Machine: overall out of sample accuracy is 0.9957 

```r
confusionMatrix(p.gbm,test$classe)  
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    2    0    0    0
##          B    0 1511    1    0    0
##          C    0    3 1359    0    0
##          D    0    2    8 1286    8
##          E    0    0    0    0 1434
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9969         
##                  95% CI : (0.9955, 0.998)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9961         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9954   0.9934   1.0000   0.9945
## Specificity            0.9996   0.9998   0.9995   0.9973   1.0000
## Pos Pred Value         0.9991   0.9993   0.9978   0.9862   1.0000
## Neg Pred Value         1.0000   0.9989   0.9986   1.0000   0.9988
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1926   0.1732   0.1639   0.1828
## Detection Prevalence   0.2847   0.1927   0.1736   0.1662   0.1828
## Balanced Accuracy      0.9998   0.9976   0.9965   0.9986   0.9972
```
2.Random Forest:overall out of sample accuracy is 0.9983 

```r
confusionMatrix(p.rf,test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    1    0    0    0
##          B    0 1516    1    0    0
##          C    0    1 1364    0    0
##          D    0    0    3 1286    2
##          E    0    0    0    0 1440
## 
## Overall Statistics
##                                          
##                Accuracy : 0.999          
##                  95% CI : (0.998, 0.9996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9987         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9987   0.9971   1.0000   0.9986
## Specificity            0.9998   0.9998   0.9998   0.9992   1.0000
## Pos Pred Value         0.9996   0.9993   0.9993   0.9961   1.0000
## Neg Pred Value         1.0000   0.9997   0.9994   1.0000   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1932   0.1738   0.1639   0.1835
## Detection Prevalence   0.2846   0.1933   0.1740   0.1645   0.1835
## Balanced Accuracy      0.9999   0.9993   0.9985   0.9996   0.9993
```

+ **Selecting final model**: based on the comparision of out of sample error, we chose the model ran by random forest, here is the model information:

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 40
## 
##         OOB estimate of  error rate: 0.1%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3348    0    0    0    0 0.0000000000
## B    0 2278    1    0    0 0.0004387889
## C    0    2 2050    2    0 0.0019474197
## D    0    0    5 1924    1 0.0031088083
## E    0    0    0    1 2164 0.0004618938
```

THE END. THANKS FOR READING！

