---
title: "Practical Machine Learning Course Project"
author: "Xiaoning PEI"
---

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
##          A 2232    6    0    0    0
##          B    0 1508    3    0    0
##          C    0    2 1357    1    0
##          D    0    2    8 1285    9
##          E    0    0    0    0 1433
## 
## Overall Statistics
##                                           
##                Accuracy : 0.996           
##                  95% CI : (0.9944, 0.9973)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.995           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9934   0.9920   0.9992   0.9938
## Specificity            0.9989   0.9995   0.9995   0.9971   1.0000
## Pos Pred Value         0.9973   0.9980   0.9978   0.9854   1.0000
## Neg Pred Value         1.0000   0.9984   0.9983   0.9998   0.9986
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1922   0.1730   0.1638   0.1826
## Detection Prevalence   0.2852   0.1926   0.1733   0.1662   0.1826
## Balanced Accuracy      0.9995   0.9965   0.9957   0.9982   0.9969
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
##          A 2232    4    0    0    0
##          B    0 1514    2    0    0
##          C    0    0 1366    1    0
##          D    0    0    0 1285    1
##          E    0    0    0    0 1441
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
## Sensitivity            1.0000   0.9974   0.9985   0.9992   0.9993
## Specificity            0.9993   0.9997   0.9998   0.9998   1.0000
## Pos Pred Value         0.9982   0.9987   0.9993   0.9992   1.0000
## Neg Pred Value         1.0000   0.9994   0.9997   0.9998   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1930   0.1741   0.1638   0.1837
## Detection Prevalence   0.2850   0.1932   0.1742   0.1639   0.1837
## Balanced Accuracy      0.9996   0.9985   0.9992   0.9995   0.9997
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
##         OOB estimate of  error rate: 0.12%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3348    0    0    0    0 0.0000000000
## B    3 2274    2    0    0 0.0021939447
## C    0    2 2050    2    0 0.0019474197
## D    0    0    2 1927    1 0.0015544041
## E    0    0    0    2 2163 0.0009237875
```

## Predicting 20 testing problems
We predicted 20 cases in testing set, and get the outcomes as following:

```r
predict(fit.rf,testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

THE END. THANKS FOR READING£¡

