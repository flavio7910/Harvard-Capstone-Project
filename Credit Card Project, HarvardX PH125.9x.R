---
title: 'Credit Card Project, HarvardX: PH125.9x'
author: "Flavio Alejandro Chavarri Miranda"
date: "12/4/2020" 
---

## 1. Excecutive Summary

# This is the final project for the HarvardX: PH125.9x Data Science: Capstone course and for the HarvardX Data Science Professional Certificate. All the courses of the professional certificate gave us the knowledge and the tools to make this capstone project. In this opportunity, I will create a model that can analyze and predict when a transaction made with a credit card was a fraud or a legal transaction.

# Following the recommendations, the dataset used in this model was downloaded from Kaggle. In Kaggle, the data was uploaded by Machine Learning Group - ULB and it is called "Credit Card Fraud Detection". The name of the excel file in a CSV format is creditcard.csv. To make it easier, I downloaded the file and uploaded to my Google Drive Account. In this project, the data is automatically downloaded from my Google Drive account, therefore there is no need to download the data from another source. The name of the dataset in this model is "creditcard". In case you want to download the data and then upload it to Rmd, you can download it from Kaggle or my Github account. Below are the links to download the data in my Google Drive account, in Kaggle, and my Github account:
  
  # https://drive.google.com/uc?id=1QpCApwEac85j2zCmcXWA03zjsZX24f7E&export=download&authuser=0 
  #https://www.kaggle.com/mlg-ulb/creditcardfraud
  #https://github.com/flavio7910/Harvard-Capstone-Project

# This dataset has a total of 284807 transactions and a total of 492 frauds. This dataset was collected from the transactions made by European credit cardholders in September 2013. Due to privacy concerns, the variables V2, V2, ... V28 does not have the names of the variables and for instance, we do not know its features. The only two variables that we have the names are the time and amount variables. 

# This model seeks to recognize when a transaction was legal or when it was a fraud. To evaluate the data, I split the data in the most common way making a 70/30 split,  70 percent for the training and 30 percent for the test set. The goal of this project is to make a model that can reach an accuracy rate above 99.95% so that we can have an almost perfect amount of true positives and true negatives in future transactions. To make this happen, I will use four models: Naive Baseline Model,Logistic Regression Model, Decision Tree Model, and Random Forest Model. 


# If necesary, install all these packages

if(!require(e1071)) install.packages("e1071")
if(!require(caret)) install.packages("caret")
if(!require(corrplot)) install.packages("corrplot")
if(!require(class)) install.packages("class")
if(!require(dplyr)) install.packages("dplyr")
if(!require(gbm)) install.packages("gbm")
if(!require(gbm2sas)) install.packages("gbm2sas")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(PRROC)) install.packages("PRROC")
if(!require(randomForest)) install.packages("randomForest")
if(!require(reshape2)) install.packages("reshape2")
if(!require(rpart.plot)) install.packages("rpart.plot")
if(!require(ROCR)) install.packages("ROCR")
if(!require(stringr)) install.packages("stringr")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse") 

# Load these libraries
library(caret)
library(caTools)
library(class)
library(dplyr)
library(e1071)
library(gbm)
library(ggplot2)
library(kableExtra)
library(PRROC)
library(randomForest)
library(readr)
library(reshape2)
library(rpart)
library(rpart.plot)
library(ROCR)
library(tidyr)
library(tidyverse)

# The data is automatically downloaded from my Google Drive account https://drive.google.com/uc?id=1QpCApwEac85j2zCmcXWA03zjsZX24f7E&export=download&authuser=0 
# After the data is downloaded, it has to be read as a csv file
creditcard2 <- tempfile()
download.file("https://drive.google.com/uc?id=1QpCApwEac85j2zCmcXWA03zjsZX24f7E&export=download&authuser=0", creditcard2)
creditcard <- read.csv(creditcard2)

## 2. Analysis and Methods

# The credit card fraud dataset has 31 colums: time, V1 to V28, amount, and class. 
# Show the top of each column
head(creditcard)

# The total length of the data is 284807. This means that in our dataset we have a total of 284807 transactions. All of the variables are numeric with the exceptions of "Class" that is an integer. 

# Show the total length of the data and the number of colums
data.frame("Length" = nrow(creditcard), "Columns" = ncol(creditcard)) %>%
  kable() 

# Show the type of data
sapply(creditcard, class)

# There are not any missing values in the columns. This will make the analysis easier. 

# Find missing values of each variable of the dataset
sapply(creditcard, function(x) sum(is.na(x))) %>% 
  kable()

# All of the variables have independent numeric values except for the class variable. The class variable is a dummy variable where 0 means that the transaction was legal, and 1 represents that the transaction was a fraud. There is a total of 284315 legal transactions and a total of 492 frauds on the data. 


# Count the amount of transactions
creditcard %>%
  group_by(Class) %>% 
  summarise(Count = n()) %>%
  kable() 

fraudvslegal <- creditcard %>%
  mutate(Class = as.factor(Class)) %>% 
  ggplot(aes(Class, fill = Class)) + labs(title = "Legal Transactions (0) vs Fraud Transactions (1)",
                                          x = "Class",
                                          y = "Frequency")+
  geom_bar()+
  theme_classic()+ 
  coord_flip()
fraudvslegal 

# Taking this fact into consideration, it is much more clear to analyze the basic summary statistics. Based on the chart, the average transaction is worth 88.35. Since the amounts are small, it is better to analyze the fraud transactions alone. 


# Show the summary statistics of each variable
summary(creditcard)

# By analyzing only the amount of the fraud transactions, we can observe that the total amount sums 60127 and the mean is 122.2113.


# First, the class 1 (Fraud) data needs to be separated
Fraudonly <- creditcard[creditcard$Class == 1, ] 

# Now we need to filter only the amount colum. With that we can get the sum and mean 
Fraudonlyamount <- Fraudonly$Amount
sum(Fraudonlyamount)
mean(Fraudonlyamount)

# Now, using only the fraud data, we can observe how many times the same amount of money was retired. Making the distribution of the transaction´s amount, we can observe that small amounts of money are more likely to be a fraud. Most of the frauds are in the first quantile as well. Below are the top 10 amounts that were a fraud. One dollar is most likely to be a fraud.  


# Frauds Amount extracted and also the frequency.
Fraudonly %>%
  ggplot(aes(Amount)) + 
  theme_classic()  +
  geom_histogram(binwidth = 40) +
  labs(title = "Frauds Distribution",
       x = "Dollars Amount",
       y = "Frequency")

# Frauds Amount extracted and also the frequency in quantile
fraudbyquantile <- melt(table(creditcard$Amount[creditcard$Class==1]))
fraudbyquantile$QuantileAmount <- cumsum(fraudbyquantile$value)
names(fraudbyquantile)[1] <- "Amount"
ggplot(fraudbyquantile[fraudbyquantile$Amount<50,], aes(x=Amount, y=QuantileAmount, color=QuantileAmount))+ 
  geom_line()+ 
  theme_classic()

# Frauds Amount extracted and also the frequency
Fraudonly %>%
  group_by(Amount) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  head(n=10) %>%
  kable() %>%
  kable_styling(font_size = 10, position = "center")

# Now, analyzing the frauds data only, we will plot a frequency distribution to see if there is a correlation between frauds and time. Watching the correlation matrix below, it is obvious that fraud does not have a correlation with time. 


# Frauds Amount extracted over it´s time distrubition. 
Fraudonly %>%
  ggplot(aes(Time)) + 
  theme_classic()  +
  geom_histogram(binwidth = 40) +
  labs(title = "Frauds Distributions with Time",
       x = "Time",
       y = "Frequency")

# Below is the correlation matrix of all of the variables. The amout variable is more correlated with V7 and V20, but there is not a significant relationship with the variables.  



# Used the corrplot to make the correlation matrix. This way it is easier to code and it looks better. 
correlations <- cor(creditcard,method="pearson")
corrplot(correlations, number.cex = .9, method = "square", type = "full", tl.cex=0.8,tl.col = "black")

## 3. Results

# Before starting to build models, the class of the credit cards has to be converted to factors in order to get the results. After that, I generated a sequence of random numbers by setting a seed. The data will be split in a train set and in a test set. The proportion of the sets is 70/30. Four models will be used: Naive Baseline Model, Logistic Regression Model, Decision Tree Model, and Random Forest Model. 

#Before continuing to build models, we have to convert the credit card class to factors
creditcard$Class <- factor(creditcard$Class)

# Generate a sequence of random numbers
set.seed(1)

# Split the data 70/30 in two sets: trainset and testset
test_index <- sample.split(creditcard$Class, SplitRatio = 0.7)
trainset <- subset(creditcard, test_index == T)
testset <- subset(creditcard, test_index == F)

### Naive Baseline Model

# The RMSE of the naive baseline model is 256.4233. The Naive model is the most simple model used since it is based on the mean. The RMSE is a prediction based on the average. The score of 256.4233 is huge, therefore it cannot indicate the performance well. For this reason, it does not make sense to analyze further the model. The other models will use accuracy as a measure of performance. 

# Calculate the average of the transactions. "Fraudonly" was coded before
meancard <- mean(Fraudonly$Amount)

# Predict the RMSE on the test set
rmsenaive <- RMSE(Fraudonly$Amount, meancard)

# Show the results
naivemodel <- data.frame(model="Naive Baseline Model", RMSE=rmsenaive)
naivemodel

### Logistic Regression Model

# The logistic regression model estimates the probability of using a cumulative logistic distribution function. This model has an accuracy of 99.900518%. The accuracy is calculated using the formula: (TP+TN)/(TP+TN+FP+FN). The results are shown in the correlation matrix.

# Call the formula of the logistic regression
regression <- glm(Class ~ ., data = trainset, family = "binomial",)

# Show the results in a confussion matrix to calculate accurracy
regressionprediction <- predict(regression, testset, type = "response")
table(testset$Class, regressionprediction > 0.5)

### Decision Tree Model

# The decision tree model makest the best split of the credit card using nodes which leads to a lower error rate. This model has an accuracy of 99.925096%. The accuracy of this model is better than the previous one, but the model can still be improved. 

# Separate the class and show the results in a tree and in a correlation matrix
decisiontree <- rpart(Class ~ ., data = trainset, method = "class", minbucket = 20)
prp(decisiontree) 
treeprediction <- predict(decisiontree, testset, type = "class")
confusionMatrix(testset$Class, treeprediction)

### Random Forest Model

# In order to use the random forest model, it is necessary to build another sequence of random numbers to build multiple decision trees and averages the results. Usually, the random forest uses 500 trees and 3 nodes, but in this case, due to the limitations of my laptop for the amount of the data, 100 trees, and 5 nodes will be used in the model. This model has an accuracy of 99.956692%. For instance, the number of true positives and true negatives are greater than the decision tree model. The model improved just a few compared to the decision tree model, but it was enough to reach the goal of getting an accuracy greater than 99.95%

# Generate a sequence of random numbers
set.seed(10)

# Show the results of the random fortest and it´s correlation matrix. I set the number of decision trees to 100 split in 5 because of the capacity of my laptop
randomforestmodel <- randomForest(Class ~ ., data = trainset,
                                  ntree = 100, nodesize = 5)
forestprediction <- predict(randomforestmodel, testset)
confusionMatrix(testset$Class, forestprediction)
varImpPlot(randomforestmodel)

## 4. Conclusion

# The goal of this project is to create a model that can examine the transactions made by the users and split them into two categories: fraud and legal transactions. This goal of this model is to achieve an accuracy rate above 99.95% using a machine learning model. After analyzing the four models above, the best model is the random forest model. It has an accuracy rate of 99.956692% which satisfies our goal and achieves the desired performance. 

# There is a couple of limitations with the data as describes before. Because of the privacy of the users, the variables V1 to V28 are unknown. For instance, I could have made a better analysis of data if I knew what each variable was. At the same time, a huge limitation that I had was my laptop. Unfortunately, I have a core I3 laptop with a low RAM capacity. Every time I had to run the model it took me more than 20 minutes. At the same time, when I run the random forest model with 500 trees and 3 nodes, the computer stop running so I had to decrease it to 100 trees and 3 nodes to analyze the results. It could be possible that by increasing the number of trees and nodes, the model could reach an accuracy level of 100%. Good future work would be to make the same model, but this time more decision trees and nodes and aim for an accuracy level of 100%. By having an accuracy level of 100%, the system would run perfectly and it is a great model to impress local banks since it is based in real transactions.