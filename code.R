##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Number of Rows and Columns
dim(edx)

# Number of unique movies
n_distinct(edx$movieId)

# The number of unique users
n_distinct(edx$userID)

#How many movie ratings are in each of the following genres in the edx dataset?
## str_detect
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

#separate_rows, much slower!
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Which movie has the greatest number of ratings?
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# What are the five most given ratings in order from most to least?
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))

# True or False: In general, half star ratings are less common than whole star ratings 
# (e.g., there are fewer ratings of 3.5 than there are ratings of 3 or 4, etc.).
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()

##########################################################
# Main Analysis for MovieLens Project
##########################################################

## Exploration of the data

### Histogram of the rating distribution
ggplot(edx, aes(rating)) + geom_histogram()

### Historgram of the ratings per user. It answers the question: Do all users left an approximately equal amount of ratings?
data.frame(userID = edx$userId, index = replicate(length(edx$userId),1)) %>%
  group_by(userID) %>%
  summarize(sum = sum(index)) %>%
  ggplot(aes(sum)) + geom_histogram() + scale_x_log10() + xlab("Rating Number in log scale") + ylab("Number of Users") + theme_classic() 

## Split the edx dataset into test and training set
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, 
                                  list = FALSE)
train <- edx[-test_index,]
test <- edx[test_index,]

test <- test %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

## Creating a function that computes the Residual Mean Squared Error (Input: True score, Predicted score)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


## Creating the final Model

### Create a sequence of lamdas from 0 to 10 in .25 steps in order to control the total variability
### of the movie effects. This means that when the sample size is really large n + lambda approximately equals n so that lamda does
### not have a great influence. Only when n is very small lamda will regulize this effect and will shrink it
### towards 0.

lambdas <- seq(0, 10, 0.25)

### To choose the best penalty term (lamda) we created various models with various lambdas.
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train$rating)
  
  # Effect of the regulized average rating of each movie
  movie_effect <- train %>% 
    group_by(movieId) %>%
    summarize(movie_effect = sum(rating - mu)/(n()+l))
  
  # Effect of the regulized average rating of each user controlled by movieID 
  user_effect <- train %>% 
    left_join(movie_effect, by="movieId") %>%
    group_by(userId) %>%
    summarize(user_effect = sum(rating - movie_effect - mu)/(n()+l))
  
  # Construction of predictors based on the model: ratings = alpha + movie ratings * x1 + user effect * x2
  predicted_ratings <- 
    test %>% 
    left_join(movie_effect, by = "movieId") %>%
    left_join(user_effect, by = "userId") %>%
    mutate(pred = mu + movie_effect + user_effect) %>%
    pull(pred)
  
  # Calculation of RMSE for each of the employed lambdas
  return(RMSE(predicted_ratings, test$rating))
})

## Create a plot of the RMSEs for the different lambdas employed
qplot(lambdas, rmses)

## Determining which lambda value results in the lowest RMSEs in the test set
lambda <- lambdas[which.min(rmses)]
lambda
## Creating the final model with the validation set using the complete edx dataset.
mu <- mean(edx$rating)

movie_effect <- edx %>% 
  group_by(movieId) %>%
  summarize(movie_effect = sum(rating - mu)/(n()+lambda))

user_effect <- edx %>% 
  left_join(movie_effect, by="movieId") %>%
  group_by(userId) %>%
  summarize(user_effect = sum(rating - movie_effect - mu)/(n()+lambda))

predicted_ratings <- validation %>% 
  left_join(movie_effect, by = "movieId") %>%
  left_join(user_effect, by = "userId") %>%
  mutate(pred = mu + movie_effect + user_effect) %>%
  pull(pred)

### Omitting NA values and creating a new dataframe with predicted ratings and ratings actually obtained
newly <- na.omit(data.frame(a = predicted_ratings, b = validation$rating))

## Final RMSE on the validation set
RMSE(newly$a, newly$b)
