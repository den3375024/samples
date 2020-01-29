# This sample of code is not intended to get some specific results, it just shows some common steps in action
# In this code we do binary classification
# Put datasource file 'toydata.csv' to your current working directory

if(!"xgboost"       %in% rownames(installed.packages())){install.packages("xgboost")}
if(!"caret"       %in% rownames(installed.packages())){install.packages("caret")}
if(!"ggplot2"       %in% rownames(installed.packages())){install.packages("ggplot2")}
if(!"readr"       %in% rownames(installed.packages())){install.packages("readr")}
if(!"dplyr"       %in% rownames(installed.packages())){install.packages("dplyr")}
if(!"data.table"       %in% rownames(installed.packages())){install.packages("data.table")}

library(data.table)
library(caret)
library(ggplot2)
library(xgboost)
library(readr)
library(dplyr)

# function to get rate of class '1'
get_class_rate <- function(dt){
  return( NROW(dt$L[dt$L==1])/nrow(dt))
}

# Getting toydata to work with
df <- fread(file = file.choose(), stringsAsFactors=FALSE, data.table=FALSE)

# Getting base info about data
nrow(df)
summary(df)

# converting column format 
df$F3 %<>% as.numeric()
summary(df$F3)
df$F5 %<>% as.numeric()
summary(df$F5)

# getting lable class distribution
table(df$L)
L_rate <- get_class_rate(df)
L_rate

# replacing NA values with mean
col_na <- colnames(df)[apply(df, 2, anyNA)]
col_na
df$F6 <- ifelse(is.na(df$F6), mean(df$F6, na.rm = TRUE), df$F6)
df$F7 <- ifelse(is.na(df$F7), mean(df$F7, na.rm = TRUE), df$F7)

# splitting data to train/test sets
smp_size <- floor(0.8 * nrow(df))
set.seed(1234)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

train <- df[train_ind, ]
test <- df[-train_ind, ]

# check if class distributions in train and test are the same as in total data
train_L_rate <- get_class_rate(train)
train_L_rate
test_L_rate <- get_class_rate(test)
test_L_rate

# getting column 'F1' values distribution
ggplot(train, aes(x=F1))+geom_histogram(binwidth=10)

# normalizing 'F1' column
trans_features <- c('F1')
train_trans <- as.data.frame(select(train, trans_features))
(trans <-  preProcess(train_trans, method = c("BoxCox", "center", "scale")) )
trans_train_values <- predict(trans, train_trans)
train[,trans_features] <- trans_train_values

ggplot(train, aes(x=F1))+geom_histogram(binwidth=0.2)
summary(train$F1)

test_trans <- as.data.frame(select(test, trans_features))
trans_test_values <- predict(trans, test_trans)
test[,trans_features] <- trans_test_values

ggplot(test, aes(x=F1))+geom_histogram(binwidth=0.2)
summary(test$F1)

#____experiment with xgboost classificator____
drops <- c('L')
params <- list(
  "objective"           = "binary:logistic"
  ,"eval_metric"         = "auc"
  ,"max_depth"           = 6
  ,"eta"                 = 0.1
)

# convert sets to matrix
X_xgboost <- NULL
X_xgboost <- xgb.DMatrix(as.matrix( train[,!names(train) %in%  drops]), label = train$L)
Y_xgboost  <- NULL
Y_xgboost  <- xgb.DMatrix(as.matrix(test[, !names(test) %in%  drops]))

# building model
model_xgb     <- NULL
model_xgb     <- xgboost(data = X_xgboost, params = params,  nrounds= 120)

# finding feature importances
importance <- NULL
importance <- xgb.importance(colnames(X_xgboost), model = model_xgb)
xgb.ggplot.importance(importance, top_n=15, n_clusters=6)

# testing model
test$probability <- NULL
test$probability <- predict(model_xgb, Y_xgboost)
level <-0.5
test$predicted  <- ifelse(test$probability > level, 1, 0)
table(test$predicted)
table(test$L)
confusionMatrix(as.factor(test$predicted), as.factor(test$L), positive = '1')

