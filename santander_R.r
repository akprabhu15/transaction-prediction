
install.packages('caret',repos = 'http://cran.rstudio.com/')
library(caret)
library(e1071)
library(ROSE)
library(dplyr)    
library(DMwR)
library(tidyr)
library(caTools)
library(Hmisc)
library(Metrics)
library(OutlierDetection)
library(randomForest)
library(caret)
library(ggplot2)

setwd("C:/Users/Ak/Desktop/santander")

train_df=read.csv("train.csv", stringsAsFactors = FALSE, header = TRUE)
test_df=read.csv("test.csv", stringsAsFactors = FALSE, header = TRUE)
train_df$target = as.factor(train_df$target)

num_features = subset(train_df, select = -c(1,2))

#outlier removal
outliers=OutlierDetection(num_features, k = 0.05 * nrow(variable1), cutoff = 0.95,
                 Method = "euclidean", rnames = FALSE, depth = FALSE,
                 dense = FALSE, distance = FALSE, dispersion = FALSE)[2]

for(i in outliers){
  train_df = train_df[-c(i),]
}

train_df = subset(train_df, select = -c(1))

#correlation analysis
corr_df=cor(num_features)
heatmap(corr_df, scale = 'row')
#no variables are correlated with each other

#split into train and test
set.seed(123)
data_model=sample.split(train_df, SplitRatio = 0.7)
data_train=subset(train_df, data_model==TRUE)
data_test=subset(train_df, data_model==FALSE)

prop.table(table(train_df$target))
#we can see that the target class is highly imbalanced
#class '0' ~91% and class '1'~9%

#feature scaling
#normality check
par(mfrow=c(3,3))
colnames<-dimnames(num_features)[[2]]
for (i in 1:200) {
  hist(num_features[,i],main = colnames[i],probability = TRUE)
}
#data is normally distributed
# feature scaling
data_train[-1]=scale(data_train[-1])
data_test[-1]=scale(data_test[-1])

table(train_df$target)
#we can see that the data is imbalanced
#balanced sampling to handle imbalanced data
data_train_sampled = ROSE(target ~ ., data = data_train, seed = 1)$data
table(data_train_sampled$target)
data_test_sampled = ROSE(target ~ ., data = data_test, seed = 1)$data
table(data_test_sampled$target)

#logistic regression
logit_model = glm(target~.,data = data_train_sampled, family = 'binomial')
logit_pred = predict(logit_model, newdata = data_test_sampled, type = 'response')
logit_pred=ifelse(logit_pred>0.5,1,0)

#Auc precision recall for logistic regression
roc.curve(data_test_sampled$target, logit_pred)
conf_matrix=confusionMatrix(factor(round(logit_pred)),factor(data_test_over$target))
recall(factor(logit_pred),factor(data_test_sampled$target))
precision(factor(logit_pred),factor(data_test_sampled$target))

#random forest
rf_model=randomForest(target ~ .,data = data_train_sampled, ntree=100)
rf_pred=predict(rf_model, data_test_sampled)

#auc recall precision for random forest
roc.curve(data_test_sampled$target, rf_pred)
conf_matrix=confusionMatrix(factor(round(rf_pred)),factor(data_test_sampled$target))
recall(factor(rf_pred),factor(data_test_sampled$target))
precision(factor(rf_pred),factor(data_test_sampled$target))


#naive bayes
nm_model = naiveBayes(target~., data=data_train_sampled)
nb_pred = predict(nm_model, data_test_sampled[,2:201], type = 'class')

#auc recall precision for naive bayes
roc.curve(data_test_sampled$target, nb_pred)
conf_matrix=confusionMatrix(factor(round(nb_pred)),factor(data_test_sampled$target))
recall(factor(nb_pred),factor(data_test_sampled$target))
precision(factor(nb_pred),factor(data_test_sampled$target))

#hence we conclude that the naive bayes gives us the highest auc precision and recall score



























