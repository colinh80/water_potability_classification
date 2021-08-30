# import libraries and data

library(tidyverse)
library(lubridate)
library(reshape2)
library(e1071)
library(class)
library(caret)

wq_df <- read.csv("/Users/colinhicks/Documents/data_science/data_sets/water_quality/water_potability.csv")

# view data

head(wq_df)

wq_df %>% summary()

# drop null values

wq_df <- wq_df %>%  na.omit()

# convert potability feature to logical

wq_df <- wq_df %>% mutate(Potability=as.logical(Potability))

# potability counts

potability_count <- wq_df %>% group_by(Potability) %>% summarise(count = n())
potability_count <- data.frame(potability_count)
potability_count
potability_bar_graph <- potability_count %>% ggplot(aes(Potability, count)) +
  geom_bar(stat="identity") +
  labs(title = "Potability Count")
potability_bar_graph

# box plots for each feature

col_names <- names(wq_df)[1:9]
Y <- wq_df$Potability %>% as.factor
boxplot.function <- function() {
  for(i in col_names) {
   plot <- ggplot(wq_df, aes_string(Y,i, group=Y)) +
     geom_boxplot() +
     labs(x = "Potability")
   print(plot)
  }
}

boxplot.function()

# correlation plot

  # melted correlations heat map
corr <- round(cor(wq_df),4)
melted_corr <- melt(corr)

ggplot(data = melted_corr, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed()

  # upper triangle of correlation matrix

upper_tri <- function(corr){
  corr[lower.tri(corr)]<- NA
  return(corr)
}

upper_tri <- upper_tri(corr)

  # upper matrix melted

melted_upper_corr <- melt(upper_tri, na.rm = TRUE)

  # cleaned correlation heat map with values

corr_heatmap <- ggplot(data = melted_upper_corr, aes(Var2, Var1, fill = value)) +
  geom_tile(color = "grey") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name="Pearson\nCorrelation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed()

corr_heatmapv2 <- corr_heatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 2) +
  theme( axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal") +
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

corr_heatmapv2
# distribution plots for each feature

wq_df %>% keep(is.numeric) %>%
  gather() %>%
  ggplot(aes(value)) +
  facet_wrap(~ key, scales ="free") +
  geom_histogram(aes(y=..density..),
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="red")


# outlier correction using inter-quartile range

outliers <- function(x) {
  
  q1 <- quantile(x, probs=.25)
  q3 <- quantile(x, probs=.75)
  iqr = q3-q1
  upper_limit = q3 + (iqr*1.5)
  lower_limit = q1 - (iqr*1.5)
  x > upper_limit | x < lower_limit
}

remove_outliers <- function(df, cols = names(df)) {
  for (col in cols) {
    df <- df[!outliers(df[[col]]),]
  }
  df
}

# counting removed outliers

nrow(wq_df) - nrow(remove_outliers(wq_df, col_names)) # 222 rows removed

# create new dataframe without outliers

wq_df_v2 <- remove_outliers(wq_df, col_names)

# compare skew adjustments
skew_old <- as.data.frame(lapply(wq_df[1:9], skewness))
skew_new <- as.data.frame(lapply(wq_df_v2[1:9], skewness))
skew_compare <- bind_rows(before = gather(skew_old), after = gather(skew_new), .id = "skew")

ggplot(skew_compare, aes(x = key, y = value, fill= skew)) +
  geom_bar(stat='identity',position=position_dodge(),
           colour="black" ) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  scale_fill_manual(values=c("red", "grey")) +
  labs(fill = "Skew Adjustment", x = "Features", y = "Skew Amount")

# distribution plots for each feature after skew adjustment

wq_df_v2 %>% keep(is.numeric) %>%
  gather() %>%
  ggplot(aes(value)) +
  facet_wrap(~ key, scales ="free") +
  geom_histogram(aes(y=..density..),
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="red")


# convert potability label to factor for predictions

wq_df_v2 <- wq_df_v2 %>% mutate(Potability = as.factor(Potability))

# create training and test sets

set.seed(355)
train_index <- createDataPartition(wq_df_v2$Potability, p = .7, list = FALSE)
train_set <- wq_df_v2[train_index,]
test_set <- wq_df_v2[-train_index,]

# normalize train and test set

data_norm <- function(x) {
  ((x - min(x))/(max(x)-min(x)))
}

train_set_norm <- as.data.frame(lapply(train_set[1:9], data_norm))
test_set_norm <- as.data.frame(lapply(test_set[1:9], data_norm))

# bind normalized training set with potability label

train_set_norm <- cbind(Potability = train_set$Potability ,train_set_norm)

# random forest prediction
set.seed(355)
rf <- train(Potability ~ ., data = train_set_norm, method = "rf" )
rf
  # predict test set using random forest

rf_fit <- predict(rf, test_set_norm)
confusionMatrix(reference = test_set$Potability, data = rf_fit, mode = "everything", 
                positive = "TRUE")

  # feature importance for random forest prediction

varimp_rf <- varImp(rf)
plot(varimp_rf, main = "Random Forest Variable Importance")

# knn prediction
set.seed(355)
knn <- train(Potability ~ ., data = train_set_norm, method = "knn" )
knn

# predict test set using random forest

knn_fit <- predict(knn, test_set_norm)
confusionMatrix(reference = test_set$Potability, data = knn_fit, mode = "everything", 
                positive = "TRUE")

# feature importance for random forest prediction

varimp_knn <- varImp(knn)
plot(varimp_knn, main = "KNN Variable Importance")
