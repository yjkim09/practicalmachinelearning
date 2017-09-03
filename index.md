# Practical Machine Learning Course Project
Chris Kim  
September 2, 2017  





## Introduction

In this project, we use the data from the accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

The goal is to predict the manner in which they did the exercise. This is given in the "classe" variable from the training set.

We make the report describing how to build the model, how to use the cross validation, and why you made the choices you did. At the end You will also use your trained model to predict 20 different test cases from the validation set.

## Data Preparation


```r
if (!file.exists("data")) {dir.create("data")}

trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(trainUrl, destfile = "./data/pml-training.csv")
inBuild <- read.table("./data/pml-training.csv",na.strings=c("","#DIV/0!"), header = TRUE, sep = ",")

testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(testUrl, destfile = "./data/pml-testing.csv")
validation <-  read.table("./data/pml-testing.csv",na.strings=c("","#DIV/0!"), header = TRUE, sep=",")
```

Testing data is named as validation because train data will be used for both training and testing, and we'll validate the accuracy of the model using the validation set.

## Preprocessing

We remove rows and columns with many **NAs**. This reduced the number of columns from 160 to 60. 


```r
# remove rows/columns with many NAs
colNAs <- apply(is.na(inBuild),MARGIN = 2,sum)
inBuild2 <- inBuild[,colNAs<10]
rowNAs <- apply(is.na(inBuild2),MARGIN = 1,sum)
inBuild2 <- inBuild2[rowNAs<50,]

sum(is.na(inBuild2))
```

```
## [1] 0
```

After this process, there is no NAs left in the new dataset. 

Now we apply **nearZeroVar** in order to remove zero covariates. Only one column is removed from this process.


```r
# Removing Zero Covariates
tmp <- nearZeroVar(inBuild2,saveMetrics = TRUE)
inBuild2 <- inBuild2[,tmp$nzv==FALSE]
```

Finally I notice by inspection that the first five columns are serial numbers, timestamps, and participant names. We'll remove them from the modeling.


```r
inBuild2 <- inBuild2[,-c(1:5)]
```

We use 60% of the data for the training and 40% for the testing.


```r
# prepare training and testing sets
set.seed(12345)
inTrain = createDataPartition(inBuild2$classe, p = 0.6, list=FALSE)
training = inBuild2[ inTrain,]
testing = inBuild2[-inTrain,]
```

- training data 

```r
dim(training)
```

```
## [1] 11776    54
```

- testing data

```r
dim(testing)
```

```
## [1] 7846   54
```

## Model

We decide to use random forecasts because it is one of the most accurate methods available along with the boosting. We use five-fold cross-validation.  


```r
# configure trainControl object (for speed)
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

fit <- train(classe~., method = "rf", data = training, trControl = fitControl, proxy = TRUE)

fit$finalModel 
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proxy = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.25%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3346    1    0    0    1 0.0005973716
## B    6 2271    2    0    0 0.0035103115
## C    0    4 2050    0    0 0.0019474197
## D    0    0    9 1920    1 0.0051813472
## E    0    0    0    6 2159 0.0027713626
```

We find from the output that the estimate of error rate is 0.25%.

## Testing


```r
pred <- predict(fit,testing)

confusionMatrix(testing$classe,pred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    2 1512    4    0    0
##          C    0    2 1366    0    0
##          D    0    0    9 1277    0
##          E    0    2    0    2 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9973          
##                  95% CI : (0.9959, 0.9983)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9966          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9974   0.9906   0.9984   1.0000
## Specificity            1.0000   0.9991   0.9997   0.9986   0.9994
## Pos Pred Value         1.0000   0.9960   0.9985   0.9930   0.9972
## Neg Pred Value         0.9996   0.9994   0.9980   0.9997   1.0000
## Prevalence             0.2847   0.1932   0.1758   0.1630   0.1833
## Detection Rate         0.2845   0.1927   0.1741   0.1628   0.1833
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9996   0.9982   0.9951   0.9985   0.9997
```

The overall accuracy rate is 99.73%.

## Validation

Now we apply the tested model to the validation set. I confirmed from the project quiz that all 20 cases are forecasted correctly.


