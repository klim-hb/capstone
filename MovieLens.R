# It is easier to follow the flow in the RMD script.

# Loading libraries
library(dplyr) #data wrangling
library(tidyverse)
library(kableExtra) #very useful package to change the style of output tables 
library(knitr)
library(tidyr)
library(stringr)
library(ggplot2)
library(recosystem) #package for recommendation system; used as second method in this report
library(tinytex) #to enable LaTeX
#library(readr) - on early stage of the project, to export data from MovieLens to csv files
                
################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

## the following line was changed from 
##    movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId)[movieID],
## in order to ensure no NAs are in movie ID and genres for this dataset. 
## we randomly checked several records and it appears that data was not compromised.
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                          title = as.character(title),
                                          genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
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

# UPDATE: the below is no longer appicable after moving to GCP:
# The original EDX code for forming the datasets EDX and VALIDATION was changed due to limitation of computing resources 
# (the R Studio and entire PC was hanging for days hence it was not possible to perform the analysis). Instead, 
# the process was divided into 2 parts: forming the datasets as per EDX instructrions and the analysis itself. 
# The files for datasets were saved separetely as csv documents and imported back to perform the analysis

#edx <- read_csv("edx.csv", col_types = cols(movieId = col_integer(), 
#                  timestamp = col_integer(), userId = col_integer()), trim_ws = FALSE)
#validation <- read_csv("validation.csv", col_types = cols(movieId = col_integer(),
#                  timestamp = col_integer(), userId = col_integer()), trim_ws = FALSE)

#installation of TinyTeX in R (was required on the first attemp to create PDF: tinytex::install_tinytex()

## RMSE

# The RMSE function in R code:
RMSE <- function(true_ratings = NULL, predicted_ratings = NULL) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
 
## Regularization

regularization <- function(lambda, trainset, testset){
                                                                                                
  # Mean
  mu <- mean(trainset$rating)
  
  # Movie effect (bi)
  b_i <- trainset %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  
  # User effect (bu)  
  b_u <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  
  # Prediction: mu + bi + bu  
  predicted_ratings <- testset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    filter(!is.na(b_i), !is.na(b_u)) %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, testset$rating))
}

## Recosystem

# There are different utilities and libraries available in R packages for testing and training the recommendation systems. In particular, recommenderlab and recosystem proved to be most efficient. 
# 
# In this project we will utilize recosystem, as it is intuitively more clear and uses simple syntax. 
# 
# recosystem is an R wrapper of the [LIBMF library](http://www.csie.ntu.edu.tw/~cjlin/libmf/) developed by [Yu-Chin Juan, Yong Zhuang, Wei-Sheng Chin and Chih-Jen Lin](http://www.csie.ntu.edu.tw/~cjlin/libmf/), an open source library for recommender system using marix factorization.
# 
# We will use this library as our second method for the project. 

## Steps of this project

# The project will be performed in the following steps:
#   
# 1. We will analyze the provided data 
# 2. We will preprocess the provided data and split edx into two parts
# 3. We will build two methods for our task: Linear Model (via RMSE and regularization) & Factorization Model (via recosystem)
# 4. We will test Linear Model against Final Validation (as a requirement of the project, the validation set could only be used once; since the aim of the project is to practice Linear Model - we will only utilize it with the LM itself.)
# 5. We will record and analyze the results 
# 6. We will conclude the discussion on both methods used with observed limitations and recommendations for future research.

                                                                                                                                                                                      # 
# Exploratory Data Analysis

## Initial Datasets

#this commented step was removed as the problem with big data processing was solved:
#exporting initial data in case R studio crashes again
#this code was commented in the final version
#write.csv(edx, "edxInitial.csv", row.names=F)
#write.csv(validation, "validationInitial.csv", row.names=F)
#Utilizing kable and kableExtra packages to format the tables for our pdf report as described here: https://cran.r-project.org/web/packages/kableExtra/vignettes/awesome_table_in_html.html
#Unfortunately, in the final pdf it does not look as neat :( 

#create a dataframe to display the basic info about both sets 
text_tbl <- data.frame(
  Feature = c("Number of Rows", "Number of Columns", "Unique Users","Unique Movies","Variety of Genres"),
  EDX = c(nrow(edx), ncol(edx),n_distinct(edx$userId),n_distinct(edx$movieId),n_distinct(edx$genres)), 
  Validation = c(nrow(validation), ncol(validation),n_distinct(validation$userId),n_distinct(validation$movieId),n_distinct(validation$genres)))

#display the dataframe with kable package, making the first row background red, font color - white and bold; 
#only first column to be also bold in values for display purposes. 

kable(text_tbl) %>%
  kable_styling(full_width = F) %>%
  column_spec(1, bold = T) %>%
  column_spec(2, bold=F) %>%
  column_spec(3, bold=F) %>%  row_spec(0, bold = T, color = "white", background = "#D7261E")

# The datasets have the following columns:
#Same as previous text_tbl - make the output neat. 

text_tbl1 <- data.frame(
  Name = c(names(edx)),
  Comment = c("Unique identification number for each user in the dataset","Unique identification number for each movie in the dataset","A range of marks(rating) given to a movie by specific user","A record of specific time when the rating was given by the user to a particular movie","Title of the movie with Release date","Genre(s) of the movie"))

kable(text_tbl1) %>%
  kable_styling(full_width = F) %>%
  column_spec(1, bold =T) %>%
  column_spec(2, bold=F) %>%
  row_spec(0, bold = T, color = "white", background = "#D7261E")

edx %>% head() %>% kable() %>%
  kable_styling(bootstrap_options = c("striped", "condensed", "responsive"),
                position = "center",
                font_size = 7,
                full_width = FALSE) %>%
  row_spec(0, bold = T, color = "white", background = "#D7261E")

# We can see the following observations:
#   
# * timestamp - the format currently displays the number of second since Jan 1, 1970 (EPOCH) and is hard to understand 
# * title - Movies' titles have the year of their release, this data might be helpful in our analysis 
# * genres - column consists of a variety of genres divided with |-sign; it might be useful to segregate the genres for our analysis
# 
# Hence, prior to starting any analysis of the data, the dataset must be put in order. 


# Methods

## Preprocessing of the data

#Preprocessing
#Please refer to the textbook section for a detailed explanation of colSds and nearZeroVar()
#no point to use because we need all columns and they were pre-selected by EDX team
#Also described here: https://topepo.github.io/caret/pre-processing.html - the type of data is really not appropriate for our purpose, we shall skipp this step and preprocess "manually"

# Convert timestamp of each rating to a proper format in EDX dataset
edx$date <- as.POSIXct(edx$timestamp, origin="1970-01-01")
# As discussed on EDX forum - we cannot change validation dataset until last RMSE function is done. This is actually a great news, since we want to keep the Global Enviroment clean 
# Convert timestamp of each rating to a proper format in validation dataset
#validation$date <- as.POSIXct(validation$timestamp, origin="1970-01-01")

# Separate the date of each rating in EDX dataset and change the format of displayed year/month/day. We want to split it because we want to dig deep into patterns between the user behavior
edx$yearR <- format(edx$date,"%Y")
edx$monthR <- format(edx$date,"%m")
edx$dayR <- format(edx$date,"%d")

# Ensure the timedate data recorded as numeric in EDX dataset
edx$yearR <- as.numeric(edx$yearR)
edx$monthR <- as.numeric(edx$monthR)
edx$dayR <- as.numeric(edx$dayR)

# Extracting the release year for each movie in edx dataset
edx <- edx %>%
mutate(title = str_trim(title)) %>%
extract(title,
c("titleTemp", "release"),
regex = "^(.*) \\(([0-9 \\-]*)\\)$",
remove = F) %>%
mutate(release = if_else(str_length(release) > 4,
as.integer(str_split(release, "-",
                 simplify = T)[1]),
as.integer(release))
) %>%
mutate(title = if_else(is.na(titleTemp),
title,
titleTemp)
) %>%
select(-titleTemp)

# An alternative solution to the above is to consider genre variability as an extension of the movie specification and 
# treat each particular combination of genres as a unique category.  
# As reported by several peers in the EDX forum, the genre specificity might have a low impact on the total RMSE,
# and hence at this stage of the project, in order to not overwhelm the environment, we will consider the alternative solution. Once the calculations for RMSE models are made, we might reconsider this step, if the RMSE values will not meet the required low values.

# only keeping what we need for EDX without any temp columns
edx <- edx %>% select(userId, movieId, rating, title, genres, release, yearR, monthR, dayR)

# After applying the above changes, the datasets have the following structure:
edx %>% head() %>% kable() %>%
  kable_styling(bootstrap_options = c("striped", "condensed", "responsive"),
                position = "center",
                font_size = 7,
                full_width = FALSE) %>%
  row_spec(0, bold = T, color = "white", background = "#D7261E")

## Arrange edx as training and test datasets as 90/10%
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = edx$rating, p=0.1, list = FALSE)
trainSet <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in validation set are also in edx set
testSet <- temp %>% 
  semi_join(trainSet, by = "movieId") %>%
  semi_join(trainSet, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, testSet)
trainSet <- rbind(trainSet, removed)
rm(test_index, temp, removed)


# Display Results
text_tbl2 <- data.frame(
  Feature = c("Number of Rows", "Number of Columns", "Unique Users","Unique Movies","Variety of Genres"),
  Train = c(nrow(trainSet), ncol(trainSet),n_distinct(trainSet$userId),n_distinct(trainSet$movieId),n_distinct(trainSet$genres)), 
  Test = c(nrow(testSet), ncol(testSet),n_distinct(testSet$userId),n_distinct(testSet$movieId),n_distinct(testSet$genres)), 
  Validation = c(nrow(validation), ncol(validation),n_distinct(validation$userId),n_distinct(validation$movieId),n_distinct(validation$genres)))

kable(text_tbl2) %>%
  kable_styling(full_width = F) %>%
  column_spec(1, bold = T) %>%
  column_spec(2, bold=F) %>%
  column_spec(3, bold=F) %>%  
  column_spec(4, bold=F) %>% row_spec(0, bold = T, color = "white", background = "#D7261E")


## Dataset analysis

### Ratings Distribution

# Heatmap for User vs Movies sample of 100
usersHM <- sample(unique(trainSet$userId), 100)
trainSet %>% filter(userId %in% usersHM) %>%
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% 
  select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")
title("User and Movie Matrix")


# create plot to show rating distribution across values(0.5-5)
trainSet %>%
  group_by(rating) %>%
  summarize(count = n()/1000) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line() + xlab("Rating 0.5-5") + ylab("# Ratings, thousands") +ggtitle("Distribution of Ratings")

# The distribution of ratings varies from user to user, movie to movie, year to year.

# let's clean the global enviroment, as we can see some smoke coming from the computer - partly solved with gooogle server, but let's still keep the enviroment clean
rm(text_tbl, text_tbl1, text_tbl2, chkEdx)

# count how many whole star ratings in the dataset
a<-sum(trainSet$rating %% 1 == 0)/length(trainSet$rating)
b<-mean(trainSet$rating)


# In general, half star ratings are less common than whole star ratings.
# In the *trainSet* we can count over 79% of whole star ratings across all users. 
# At the same time, the avarage ratings across the dataset is 3.5.

# show rating distribution over years; Im assigning to ratdistY because otherwise shows warning for geom_path
trainSet %>%
  group_by(yearR,rating) %>%
  summarize(count = n()/1000) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line() + xlab("Rating scale") + ylab("# Ratings, thousands") +
  ggtitle("Distribution of Ratings over Years")+facet_wrap(~yearR, nrow=3) + 
  theme(axis.text.x=element_text(size=7, vjust=0.5))

#ReleaseRating Plot
trainSet %>% group_by(release) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(release, rating)) +
  geom_point() +
  geom_smooth()

# show rating distribution over months over years
trainSet %>%
  group_by(yearR,monthR) %>%
  summarize(count = n()/1000) %>%
  ggplot(aes(x = monthR, y = count)) +
  geom_point() +
  scale_x_continuous(name="Months",breaks=seq(1,12,1),
                     labels=c("Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"))+
  ylab("# Ratings, thousands") +ggtitle("Distribution of Ratings over Months") +
  facet_wrap(~yearR, nrow=3) + theme(axis.text.x=element_text(angle=90, size=7, vjust=0.5))

# show rating distribution over days
trainSet %>%
  group_by(dayR, monthR) %>% #filter(yearR==1996) %>%
  summarize(count = n()/1000) %>%
  ggplot(aes(x = dayR, y = count)) +
  geom_point() +
  scale_x_continuous(name="Days",breaks=seq(1,31,1))+
  theme(axis.text.x = element_text(size=7, angle=45))+
  ylab("# Ratings, thousands") +ggtitle("Distribution of Ratings over Days for all Years") +
  facet_wrap(~monthR, nrow=3) + theme(axis.text.x=element_text(angle=90, size=7, vjust=1))

# show rating distribution over release years
trainSet %>%
  group_by(release,rating) %>%
  summarize(count = n()/1000) %>%
  ggplot(aes(x = release, y = count)) +
  geom_line() + xlab("Year of Release") + ylab("# Ratings, thousands") +
  ggtitle("Distribution of Ratings over Release Years")+facet_wrap(~rating, nrow=2) + 
  theme(axis.text.x=element_text(size=7, vjust=0.5))

# show rating distribution over movies over years
trainSet %>%
  group_by(movieId, yearR) %>%
  summarize(count = n()/1000) %>%
  ggplot(aes(x = movieId, y = count)) +
  geom_line() + xlab("Movie ID") + ylab("# Ratings, thousands") +
  ggtitle("Distribution of Ratings over Movies over Years")+facet_wrap(~yearR, nrow=3) + 
  theme(axis.text.x=element_text(size=7, vjust=0.5))

# show rating distribution over movies over years
trainSet %>%
  group_by(movieId, release) %>%
  summarize(count = n()/1000) %>%
  ggplot(aes(x = movieId, y = count)) +
  geom_line() + xlab("Movie ID") + ylab("# Ratings, thousands") +
  ggtitle("Distribution of Ratings over Movies over Release Years")+
  facet_wrap(~release, nrow=7) + theme(axis.text.x=element_text(size=0, vjust=0.5))

# To understand the data origin, we need to examine how users rated movies in the datasets.

trainSet %>% group_by(userId) %>%
  summarise(count=n()) %>%
  ggplot(aes(count)) +
  geom_histogram(color = "white") +
  scale_x_log10() + 
  ggtitle("Distribution of Users") +
  xlab("Number of Ratings") +
  ylab("Number of Users") 

# Genres popularity dataset from trainset
genres_popularity <- trainSet %>%
  na.omit() %>% # omit missing values
  select(movieId, yearR, genres) %>% # select columns we are interested in
  mutate(genres = as.factor(genres)) %>% # turn genres in factors
  group_by(yearR, genres) %>% # group data by year and genre
  summarise(number = n()) 

# Selective genres vs year of rating
genres_popularity %>%
  filter(yearR > 1996) %>%
  filter(genres %in% c("Action","Adventure","Animation","Children","Comedy","Fantasy","Drama","Thriller", "Romance","War", "Sci-Fi", "Western")) %>%
  ggplot(aes(x = yearR, y = number)) +
  geom_line(aes(color=genres)) +
  scale_fill_brewer(palette = "Paired") 

# Results

## Linear Model
### Mean distribution model

# Mean of observed values
mu <- mean(trainSet$rating)
mu

# calculate RMSE
naive_rmse <- RMSE(testSet$rating,mu)

# show results table with the findings from the first model  
results <- tibble(Method = "Mean", RMSE = naive_rmse, Tested = "testSet")

results %>% kable() %>%
kable_styling(bootstrap_options = c("striped", "condensed", "responsive"),
position = "center",
font_size = 9,
full_width = FALSE) %>%
row_spec(0, bold = T, color = "white", background = "#D7261E")

### Movie-centered model (Adding Movie Effects)

# Define movie effect b_i
bi <- trainSet %>% 
group_by(movieId) %>% 
summarize(b_i = mean(rating - mu))

# Rating with mean + b_i  
y_hat_bi <- mu + testSet %>% 
left_join(bi, by = "movieId") %>% 
.$b_i

# Calculate the RMSE  
results <- bind_rows(results, tibble(Method = "Mean + bi", RMSE = RMSE(testSet$rating, y_hat_bi), Tested = "testSet"))

# Show the RMSE improvement  

results %>% kable() %>%
kable_styling(bootstrap_options = c("striped", "condensed", "responsive"),
            position = "center",
            font_size = 9,
            full_width = FALSE) %>%
row_spec(0, bold = T, color = "white", background = "#D7261E")


### Movie + User model (Adding User Effects)


#Plot for user effect distribution
trainSet %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(color = "black") + 
  ggtitle("User Effect Distribution") +
  xlab("User Bias") +
  ylab("Count")


#Apply the model        
# User effect (bu)
bu <- trainSet %>% 
  left_join(bi, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Prediction
y_hat_bi_bu <- testSet %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# Calculate the RMSE  
results <- bind_rows(results, tibble(Method = "Mean + bi + bu", RMSE = RMSE(testSet$rating, y_hat_bi_bu), Tested = "testSet"))

# Show the RMSE improvement  

results %>% kable() %>%
  kable_styling(bootstrap_options = c("striped", "condensed", "responsive"),
                position = "center",
                font_size = 9,
                full_width = FALSE) %>%
  row_spec(0, bold = T, color = "white", background = "#D7261E")
              

### Movie + User + Genre model
              
# it is assumed that adding genre to the model will not make a significant impact
# on the RMSE performance, due to the fact that genres are now not separated into their groups,
# are repetitive by nature and consume many computing resources to be correctly calculated.
# We will not include this model unless absolutely necessary (if RMSE will not reach the desired value). 
                
                
### Regularization
# regularization must be performed in order to improve the RMSE results.
 # Define a set of lambdas to tune
 lambdas <- seq(0, 10, 0.25)
 
 # Tune lambda
 rmses <- sapply(lambdas, 
                 regularization, 
                 trainset = trainSet, 
                 testset = testSet)

 
 # We can construct a plot to find out. 

 # Plot the lambda vs RMSE
 tibble(Lambda = lambdas, RMSE = rmses) %>%
   ggplot(aes(x = Lambda, y = RMSE)) +
   geom_point() +
   ggtitle("Regularization") 
 
 # We pick the lambda that returns the lowest RMSE.
 lambda <- lambdas[which.min(rmses)]
 lambda

 # We will apply this parameter to our model:
 # Then, we calculate the predicted rating using the best parameters 
 # achieved from regularization.  
 mu <- mean(trainSet$rating)
 
 # Movie effect (bi)
 b_i <- trainSet %>% 
   group_by(movieId) %>%
   summarize(b_i = sum(rating - mu)/(n()+lambda))
 
 # User effect (bu)
 b_u <- trainSet %>% 
   left_join(b_i, by="movieId") %>%
   group_by(userId) %>%
   summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

 # Prediction
 y_hat_reg <- testSet %>% 
   left_join(b_i, by = "movieId") %>%
   left_join(b_u, by = "userId") %>%
   mutate(pred = mu + b_i + b_u) %>%
   pull(pred)

 # Update the result table
 results <- bind_rows(results, 
                      tibble(Method = "Regularized bi and bu", 
                             RMSE = RMSE(testSet$rating, y_hat_reg),
                             Tested  = "testSet"))
 
 # Show the RMSE improvement  
 
 results %>% kable() %>%
   kable_styling(bootstrap_options = c("striped", "condensed", "responsive"),
                 position = "center",
                 font_size = 9,
                 full_width = FALSE) %>%
   row_spec(0, bold = T, color = "white", background = "#D7261E")
 

# We can now see significant improvement and potentially meeting the requirement for the project. 
# We should still use this model on the final validation set to approve the result. 


# ## Factorization Model

# ### Recosystem

# We followed the steps described in the package documentation, and it took approximately an hour for the script to be
# executed. However, the results were worth waiting: 

set.seed(123, sample.kind = "Rounding") # This is a randomized algorithm
# Convert the train and test sets into recosystem input format
train_data <-  with(trainSet, data_memory(user_index = userId,
                                           item_index = movieId,
                                            rating     = rating))
 test_data  <-  with(testSet,  data_memory(user_index = userId,
                                           item_index = movieId,
                                           rating     = rating))
# Create the model object
r <-  recosystem::Reco()
# Select the best tuning parameters
 opts <- r$tune(train_data, opts = list(dim = c(10, 20, 30),
                                        lrate = c(0.1, 0.2),
                                        costp_l2 = c(0.01, 0.1),
                                        costq_l2 = c(0.01, 0.1),
                                        nthread  = 4, niter = 10))
# Train the algorithm
 r$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))


### Results


# Calculate the predicted values  
y_hat_reco <-  r$predict(test_data, out_memory())
head(y_hat_reco, 10)
 
# # Update the result table
results <- bind_rows(results, 
                    tibble(Method = "Recosystem", 
                    RMSE = RMSE(testSet$rating, y_hat_reco),
                   Tested  = "testSet"))

# Show the RMSE improvement  
 
 results %>% kable() %>%
  kable_styling(bootstrap_options = c("striped", "condensed", "responsive"),
                 position = "center",
                font_size = 9,
                full_width = FALSE) %>%
   row_spec(0, bold = T, color = "white", background = "#D7261E")



# Validation and Final Results

## Validation

# Validation was performed on the *validation* set using the LM prediction after regularising the data:
mu_edx <- mean(edx$rating)

# Movie effect (bi)
b_i_edx <- edx %>% 
group_by(movieId) %>%
summarize(b_i = sum(rating - mu_edx)/(n()+lambda))

# User effect (bu)
b_u_edx <- edx %>% 
left_join(b_i_edx, by="movieId") %>%
group_by(userId) %>%
summarize(b_u = sum(rating - b_i - mu_edx)/(n()+lambda))

# Prediction
y_hat_edx <- validation %>% 
left_join(b_i_edx, by = "movieId") %>%
left_join(b_u_edx, by = "userId") %>%
mutate(pred = mu_edx + b_i + b_u) %>%
pull(pred)

# Update the result table
results <- bind_rows(results, 
                   tibble(Method = "Regularized EDX's bi and bu on Validation Set", 
                          RMSE = RMSE(validation$rating, y_hat_edx),
                          Tested  = "validation"))

# Show the RMSE improvement  < 0.86490 

results %>% kable() %>%
kable_styling(bootstrap_options = c("striped", "condensed", "responsive"),
              position = "center",
              font_size = 9,
              full_width = FALSE) %>%
row_spec(0, bold = T, color = "white", background = "#D7261E")


## Final Results Table

# The below table summarizes the final results for the Models: 

# Show the final Results table RMSE improvement  < 0.86490 
results %>% kable() %>%
kable_styling(bootstrap_options = c("striped", "condensed", "responsive"),
              position = "center",
              font_size = 9,
              full_width = FALSE) %>%
row_spec(0, bold = T, color = "white", background = "#D7261E")


# Conclusion

## Results

# Show the final Results table RMSE improvement  < 0.86490 
results %>% kable() %>%
kable_styling(bootstrap_options = c("striped", "condensed", "responsive"),
              position = "center",
              font_size = 9,
              full_width = FALSE) %>%
row_spec(0, bold = T, color = "white", background = "#D7261E")

# Session Info
sessionInfo()
                                                                                                                                                                                                                                                                                                                        ```
                                                                                                                                                                                                                                                                                                                                                                      