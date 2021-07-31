# install.packages("glmnet")
# install.packages("e1071")
# install.packages("klaR")
# install.packages("kernlab")
# install.packages("ellipse")
# install.packages("randomForest")

library(caret)
# Loading the dataset, Please draw attention to the given path for the dataset.
dataset_filename <- "C:\\edx\\Car_dataset.csv"

# load the dataset file in csv format.
car_dataset <- read.csv(dataset_filename, header = FALSE)


car_dataset[, 7] <-  as.factor(car_dataset[, 7])


# name the columns of the dataset.
colnames(car_dataset) <-
  c(
    "buying_price",
    "maintenance_price",
    "Number_of_doors",
    "persons_capacity",
    "lug_boot_size",
    "safety",
    "class"
  )





# Divide the dataset into 80% for modelling (training and testing), and 20% for validation.
set.seed(5)
holdoutIndex <-
  createDataPartition(car_dataset$class, p = 0.80, list = FALSE)
validation <- car_dataset[-holdoutIndex, ]
edx <- car_dataset[holdoutIndex, ]


# Get the dimensions of dataset.
dim(edx)

# Have a look to the dataset
head(edx, n = 20)

# Check data types of the dataset attributes.
sapply(edx, class)

# Summarize the dataset
summary(edx)

# get multiple class labels
levels(edx$class)


# check class distribution
cbind(freq = table(edx$class),
      percentage = prop.table(table(edx$class)) * 100)


# check correlations between input attributes
cases <- complete.cases(edx)
cor(edx[cases, 1:6])

# show histograms for each input variable
par(mfrow = c(2, 3))
for (i in 1:6) {
  hist(edx[, i], main = names(edx)[i])
}


# show density plot for each input variable
par(mfrow = c(2, 3))
cases <- complete.cases(edx)
for (i in 1:6) {
  plot(density(edx[cases, i]), main = names(edx)[i])
}


# show box plots for each input variable
par(mfrow = c(2, 3))
for (i in 1:6) {
  boxplot(edx[, i], main = names(edx)[i])
}




# show bar plots of each attribute by class
par(mfrow = c(2, 3))
for (i in 1:6) {
  barplot(
    table(edx$class, edx[, i]),
    main = names(edx)[i],
    legend.text = unique(edx$class)
  )
}


## Use 10-fold cross validation with 5 repeats as resampling technique
train_resampling <-
  trainControl(method = "repeatedcv",
               number = 10,
               repeats = 5)
evaluation_metric <- "Accuracy"

# Linear Discriminate Analysis
set.seed(5)
fit.lda <-
  train(
    class ~ .,
    data = edx,
    method = "lda",
    metric = evaluation_metric,
    preProc = c("BoxCox"),
    trControl = train_resampling
  )

# Regularized Logistic Regression
set.seed(5)
fit.glmnet <-
  train(
    class ~ .,
    data = edx,
    method = "glmnet",
    metric = evaluation_metric,
    preProc = c("BoxCox"),
    trControl = train_resampling
  )


# k-Nearest Neighbors
set.seed(5)
fit.knn <-
  train(
    class ~ .,
    data = edx,
    method = "knn",
    metric = evaluation_metric,
    preProc = c("BoxCox"),
    trControl = train_resampling
  )


# Classification and Regression Trees
set.seed(5)
fit.cart <-
  train(
    class ~ .,
    data = edx,
    method = "rpart",
    metric = evaluation_metric,
    preProc = c("BoxCox"),
    trControl = train_resampling
  )


# Support Vector Machine
set.seed(5)
fit.svm <-
  train(
    class ~ .,
    data = edx,
    method = "svmRadial",
    metric = evaluation_metric,
    preProc = c("BoxCox"),
    trControl = train_resampling
  )

# Random Forest
set.seed(5)
fit.rf <-
  train(
    class ~ .,
    data = edx,
    method = "rf",
    metric = evaluation_metric,
    preProc = c("BoxCox"),
    trControl = train_resampling
  )

#   algorithm Comparison
training_results <-
  resamples(
    list(
      Algorithm_LDA = fit.lda,
      Algorithm_GLMNET = fit.glmnet,
      Algorithm_KNN = fit.knn,
      Algorithm_CART = fit.cart,
      Algorithm_SVM = fit.svm,
      Algorithm_rf = fit.rf
    )
  )
summary(training_results)
dotplot(training_results)

# print the Best Model
print(fit.rf)


# doing data transform parameters
set.seed(5)

edxx <- edx[, 1:6]
Params <- preProcess(edxx, method = c("BoxCox"))
x <- predict(Params, edxx)


validationv <- predict(Params, validation[, 1:6])

# make predictions On unseen data
set.seed(5)
preds <- predict(fit.rf, validation)
confusionMatrix(preds, validation$class)
