# Amazon Employee Access Competition

library(tidymodels)
library(dplyr)
library(vroom)
library(embed)

trainData <- vroom("amazon_train.csv")
testData <- vroom("amazon_test.csv")

plot1 <- ggplot(data = trainData, aes(x = factor(ROLE_FAMILY))) +
  geom_bar() +
  xlab("Department") +
  ylab("Count")

plot1

role_counts <- trainData %>%
  group_by(ROLE_TITLE) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

role_counts <- role_counts %>%
  mutate(percentage = count / sum(count))

role_counts

top_5_roles <- role_counts %>%
  top_n(5, wt = count)

top_5_roles

plot2 <- ggplot(top_5_roles, aes(x = reorder(ROLE_TITLE, count), y = count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +  # Flip the coordinates for horizontal bars
  labs(title = "Top 5 Role Titles by Count", x = "Role Title", y = "Count") +
  theme_minimal()

plot2
  
  
my_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_predictors(), threshold = .001) %>%
  step_dummy(all_predictors()) 

prep <- prep(my_recipe)
baked <- bake(prep, new_data = trainData)

baked
  