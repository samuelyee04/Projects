
# ----- Part 1: Data Cleaning ------

# (a) Remove Duplicate rows
# (b) Columns Selection
# (c) Replace Blank Strings with NA
# (d) Impute Missing Values
# (e) Convert Datatype to Factor
# (f) One-Hot Encoding

# 1. Denoising Autoencoder with Mode Imputation for categorical
# 2. Random Forest Imputation for numerical

# -----------------------------------------------------------------------------

# ***** 1. Denoising Autoencoder with Mode Imputation for categorical *****
# Load the required libraries
library(dplyr)
library(caret)
library(keras)
library(tensorflow)

# Install Keras from Python
keras::install_keras()

# Install Tensor Flow from Python
tensorflow::install_tensorflow()

# Check current Python file path
reticulate::py_config()

# Do this if you got warning message or permission issues
# Manually set the file path (Use version 3.10 to resolve compatible issue with Tenderflow in Keras)
reticulate::use_python("C:\\Users\\samue\\AppData\\Roaming\\uv\\python\\cpython-3.10.19-windows-x86_64-none\\python.exe", required = TRUE)

# Load dataaset
retail_data <- read.csv("C:\\Users\\samue\\Downloads\\Year 2 Sems 1 Assignments\\Programming for Data Analysis (PFDA)\\retail_data 1 (1).csv", stringsAsFactors = FALSE)

# Check total number of duplicate rows
sum(duplicated(retail_data))

# Show duplicate rows and columns
retail_data[duplicated(retail_data),]

# Remove them
retail_data <- retail_data[!duplicated(retail_data),]

# Select 10 categorical variables for cleaning
categorical_vars <- c("Income", "Customer_Segment", "Product_Category",
                      "Product_Brand", "Product_Type", "Shipping_Method",
                      "Payment_Method", "Order_Status", "Ratings", "products")

# Filter them
retail_categorical <- retail_data %>%
  select(any_of(categorical_vars))

# Details of selected columns
glimpse(retail_categorical)

# Convert empty strings to NA before converting to factor
retail_categorical[retail_categorical == ""] <- NA

# Check number of NA for each column
colSums(is.na(retail_categorical))

# Convert all columns to factor
for (col_name in names(retail_categorical)) {
  if (!is.factor(retail_categorical[[col_name]])) {
    retail_categorical[[col_name]] <- as.factor(retail_categorical[[col_name]])
  }
}

# Updated data type
glimpse(retail_categorical)

# Prepare Data for Denoising Autoencoder 
# Fill up NA with mode for trianing

# Define function for mode
get_mode <- function(v) {
  v_non_na <- v[!is.na(v)]
  if (length(v_non_na) == 0) {
    return(NA_character_) # Return NA_character_ for empty vectors after removing NA
  }
  uniqv <- unique(v_non_na)
  uniqv[which.max(tabulate(match(v_non_na, uniqv)))]
}

# Store cleaned result
retail_categorical_complete <- retail_categorical

for (col_name in names(retail_categorical_complete)) {
  if (any(is.na(retail_categorical_complete[[col_name]]))) {
    mode_value <- get_mode(retail_categorical_complete[[col_name]])
    if (!is.na(mode_value)) { # Only impute if a mode was successfully found
      retail_categorical_complete[[col_name]][is.na(retail_categorical_complete[[col_name]])] <- mode_value
    } else {
      warning(paste0("Column '", col_name, "' has no non-NA values to calculate a mode. Imputation skipped for this column in target data."))
    }
  }
}

# Verify the target data has no NA in the selected columns
print(sapply(retail_categorical_complete, function(x) sum(is.na(x))))


# One-Hot Encode for NA dataset and cleaned dataset
dummy_model <- dummyVars(~ ., data = retail_categorical_complete, fullRank = FALSE)

# Encode cleaned dataset
x_train_complete_ohe <- predict(dummy_model, newdata = retail_categorical_complete)
x_train_complete_ohe <- as.matrix(x_train_complete_ohe)

# Encode NA dataset
x_train_incomplete_ohe <- predict(dummy_model, newdata = retail_categorical)
x_train_incomplete_ohe <- as.matrix(x_train_incomplete_ohe)

# Convert NA in the incomplete OHE data to 0
x_train_incomplete_ohe[is.na(x_train_incomplete_ohe)] <- 0

# Check dimension of both encoded dataset (Make sure they are matched)
# Encoded cleaned dataset
print(paste(dim(x_train_complete_ohe), collapse = " x "))

# Encoded NA dataset
print(paste(dim(x_train_incomplete_ohe), collapse = " x "))

# Build model
input_dimension <- ncol(x_train_complete_ohe)

# Define the architecture of the autoencoder (encoder and decoder layers)
encoder_layer_1_dim <- floor(input_dimension * 0.75)
latent_space_dim <- floor(input_dimension * 0.5) # The compressed representation layer
decoder_layer_1_dim <- encoder_layer_1_dim

model <- keras_model_sequential() %>%
  # Encode
  layer_dense(units = encoder_layer_1_dim, activation = "relu", input_shape = input_dimension, name = "encoder_layer_1") %>%
  layer_dense(units = latent_space_dim, activation = "relu", name = "latent_space_layer") %>%
  
  # Decode
  layer_dense(units = decoder_layer_1_dim, activation = "relu", name = "decoder_layer_1") %>%
  layer_dense(units = input_dimension, activation = "sigmoid", name = "output_layer") # Sigmoid for probabilities 0 to 1

# Compile model
model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "binary_crossentropy" # Great for categorical variables
)

print(summary(model))

# Train model
history <- model %>% fit(
  x = x_train_incomplete_ohe, # Input data with "missingness" (0s for NAs)
  y = x_train_complete_ohe,   # Target data (mode-imputed)
  epochs = 50,                # Number of training iterations (adjust for convergence)
  batch_size = 32,            # Number of samples per gradient update
  validation_split = 0.2,     # Hold out 20% of data for validation during training
  callbacks = list(
    # Early stopping helps prevent overfitting by monitoring validation loss
    callback_early_stopping(patience = 10, monitor = "val_loss", restore_best_weights = TRUE),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 5)
  )
)

# Plot the training history to visualize loss and validation loss over epochs
plot(history)

# Imputation with trained model
# Predict probabilities for the reconstructed data
imputed_ohe_data_probs <- predict(model, x_train_incomplete_ohe)

# Keras's predict() might strip dimnames, causing errors in subsequent subsetting.
colnames(imputed_ohe_data_probs) <- colnames(x_train_complete_ohe)


# Reverse One-Hot Encoding and Populate Imputed Data
# Create a copy of the original categorical data structure to store the imputed results
retail_categorical_imputed <- retail_categorical

# Get the names of the one-hot encoded columns (used for mapping back to original categories)
ohe_column_names_map <- colnames(x_train_complete_ohe)

# Iterate through each of the original categorical variables
for (var_name in names(retail_categorical)) {
  # Find all OHE columns that correspond to the current original variable
  # Example: "Income.High", "Income.Medium", "Income.Low" for "Income"
  corresponding_ohe_cols <- grep(paste0("^", var_name, "\\."), ohe_column_names_map, value = TRUE)
  
  # Handle cases where dummyVars might not add a ".Level" suffix (binary variables)
  if (length(corresponding_ohe_cols) == 0) {
    if (var_name %in% ohe_column_names_map) {
      corresponding_ohe_cols <- var_name # It's a single OHE column (binary data)
    } else {
      warning(paste("Could not find corresponding OHE columns for variable:", var_name, ". Skipping this variable for reverse encoding."))
      next # Skip to the next variable if OHE mapping is unclear
    }
  }
  
  # Extract the predicted probabilities for the current variable's OHE columns
  probs_for_current_var <- imputed_ohe_data_probs[, corresponding_ohe_cols, drop = FALSE]
  
  # Identify rows in the original data that had missing values for this variable
  rows_to_impute <- which(is.na(retail_categorical[[var_name]]))
  
  if (length(rows_to_impute) > 0) {
    # For each row that originally had a missing value:
    for (row_idx in rows_to_impute) {
      # Get the predicted probabilities for this specific row and current variable
      current_row_probs <- probs_for_current_var[row_idx, ]
      
      # Find the index of the category with the highest predicted probability
      max_prob_idx <- which.max(current_row_probs)
      
      if (length(max_prob_idx) == 0 || is.na(max_prob_idx)) {
        # Fallback if no clear maximum is found (all probabilities are 0 or NA)
        warning(paste("No clear max probability for row", row_idx, "variable", var_name,
                      ". Defaulting to mode from the complete set for this instance."))
        retail_categorical_imputed[row_idx, var_name] <- retail_categorical_complete[row_idx, var_name]
      } else {
        # Get the full OHE column name of the most likely category ("Income.High")
        most_likely_ohe_col_name <- corresponding_ohe_cols[max_prob_idx]
        
        # Extract the original category name from the OHE column name ("High")
        # Use 'sub' to remove the variable name and "." prefix
        imputed_category_name <- sub(paste0("^", var_name, "\\."), "", most_likely_ohe_col_name)
        
        # Assign the determined category back to the imputed data frame
        retail_categorical_imputed[row_idx, var_name] <- imputed_category_name
      }
    }
  }
}

# Final check
# Verify that NA are imputed in the result for the processed columns

# Check NA after using trained model to impute data
print(sapply(retail_categorical_imputed, function(x) sum(is.na(x))))

# Compare both
View(retail_categorical)
View(retail_categorical_imputed)

# Check specific variables (frequencies before vs. after imputation)
print(table(retail_categorical$Customer_Segment, useNA = "ifany"))
print(table(retail_categorical_imputed$Customer_Segment, useNA = "no"))

# Integrate the imputed categorical data back into the full original dataset
# Replace original categorical columns with NA with imputed version
retail_final_imputed_full <- retail_data
# Ensure column names match for direct assignment
if (all(names(retail_categorical_imputed) %in% names(retail_final_imputed_full))) {
  retail_final_imputed_full[, names(retail_categorical_imputed)] <- retail_categorical_imputed
} else {
  warning("Some imputed categorical columns do not exist in the full original 'retail_data'. Cannot integrate all imputed columns directly.")
  # If this warning appears, you might need to manually merge or check column names
}

print(head(retail_final_imputed_full))

# Check NA for selected columns
retail_final_imputed_full[retail_final_imputed_full == ""] <- NA
colSums(is.na(retail_final_imputed_full))

# Save the final complete dataset
write.csv(retail_final_imputed_full, "retail_data_imputed_denoising_autoencoder.csv", row.names = FALSE)

# -----------------------------------------------------------------------------



# *****  Random Forest Imputation for numerical *****

# Install required libraries
library(missForest)
library(dplyr)
library(ggplot2)
library(tidyr)

# Load cleaned dataset
retail_data <- read.csv("C:\\Users\\ASUS\\Documents\\retail_data_imputed_denoising_autoencoder.csv", stringsAsFactors = FALSE)

# Preprocess Data for MissForest
# Define categorical variables to ensure they are handled as factors.
# These cleaned columns will be used as predictors by MissForest when imputing Total_Purchases.
categorical_imputed <- c("Total_Purchases", "Ratings", "Income", "Customer_Segment") # Only other 3 columsn are selected to reduce computational time


retail_data_imputed <-retail_data[, categorical_imputed]

unique(retail_data_imputed$Total_Purchases)

# Process all columns: There is no need to convert blank strings to NA
# Since the columns are cleaned and NA is identifiedi in Total_Puchases
for (col_name in names(retail_data_imputed)) {
  # Ensure categorical columns are factors
  if (col_name %in% categorical_imputed) {
    if (!is.factor(retail_data_imputed[[col_name]])) {
      retail_data_imputed[[col_name]] <- as.factor(retail_data_imputed[[col_name]])
    }
  }
  # Ensure Total_Purchases is numeric
  if (col_name == "Total_Purchases") {
    if (!is.numeric(retail_data_imputed[[col_name]])) {
      retail_data_imputed[[col_name]] <- as.numeric(retail_data_imputed[[col_name]])
    }
  }
}

# Check NA and datatype for Total_Purchases
glimpse(retail_data_imputed)
table(retail_data_imputed$Total_Purchases)
colSums(is.na(retail_data_imputed))

# Implement MissForest Imputation
set.seed(42) # For reproducibility of the imputation results

numerical_cols_imputed <- missForest(retail_data_imputed,
                                          maxiter = 10,  # Number of iterations
                                          ntree = 100,   # Number of trees in each forest
                                          verbose = TRUE) # Display progress during imputation

# The imputed data is available in the '$ximp' component of the returned object.
result <- numerical_cols_imputed$ximp

# The '$OOBerror' provides imputation error estimates:
# NRMSE for numerical variables, PFC for categorical variables.
print(numerical_cols_imputed$OOBerror)
# Look specifically for NRMSE for 'Total_Purchases' if it was imputed.

# Verify Imputation and Inspect Results for 'Total_Purchases'
print(sapply(retail_data_imputed, function(x) sum(is.na(x))))

print(sapply(result, function(x) sum(is.na(x))))

print(summary(retail_data_imputed$Total_Purchases))

print(summary(result$Total_Purchases))

unique(retail_data_imputed$Total_Purchases)

unique(result$Total_Purchases)

# Visual comparison of 'Total_Purchases' distribution before and after imputation
# Create a data frame for plotting
plot_df <- data.frame(
  Total_Purchases = c(retail_data_imputed$Total_Purchases, result$Total_Purchases),
  Source = c(rep("Original Data", nrow(retail_data_imputed)), rep("Imputed Data", nrow(result)))
)

# Filter out NAs from original data for plotting its distribution
plot_df_filtered <- plot_df %>%
  filter(!is.na(Total_Purchases))

# Density plot for Total_Purchases
p <- ggplot(plot_df_filtered, aes(x = Total_Purchases, fill = Source)) +
  geom_density(alpha = 0.5) +
  labs(title = "Before vs. After Imputation",
       x = "Total Purchases",
       y = "Density",
       fill = "Data Source") +
  theme_minimal()
print(p)

# Replace original categorical columns with NA with imputed version
retail_final_imputed_full_v2 <- retail_data
# Ensure column names match for direct assignment
if (all(names(result) %in% names(retail_final_imputed_full_v2))) {
  retail_final_imputed_full_v2[, names(retail_data)] <- result
} else {
  warning("Some imputed categorical columns do not exist in the full original 'retail_data'. Cannot integrate all imputed columns directly.")
  # If this warning appears, you might need to manually merge or check column names
}

glimpse(retail_final_imputed_full_v2)

unique(retail_final_imputed_full_v2$Total_Purchases)

# Save the Imputed Data 
write.csv(retail_final_imputed_full_v2, "retail_data_imputed_missforest_total_purchases_focused.csv", row.names = FALSE)

# ------------------------------------------------------------------------------



# Objective 1: To evaluate the correlation between product category with customers' ratings.
# Samuel Yee Jian Hung TP073961

# Load libraries
library(tidyverse) 
library(ranger)    
library(pdp)       
library(DALEX)     
library(caret)     
library(ROCR)      

# Load the dataset
retail_data <- read.csv("retail_data_imputed_missforest_total_purchases_focused.csv")

# Select relevant columns and convert to appropriate types
# Ensure 'Low' comes before 'High' for correct ordering in Ratings factor
selected_data <- retail_data %>%
  select(Income, Customer_Segment, Total_Purchases, Product_Category,
         Shipping_Method, Payment_Method, Order_Status, Ratings) %>%
  mutate(
    # Convert independent categorical variables to factors
    Income = factor(Income, levels = c("Low", "Medium", "High"), ordered = TRUE),
    Customer_Segment = as.factor(Customer_Segment),
    Product_Category = as.factor(Product_Category),
    Shipping_Method = as.factor(Shipping_Method),
    Payment_Method = as.factor(Payment_Method),
    Order_Status = as.factor(Order_Status),
    # Convert Ratings to an ordered factor (binary outcome: Low/High)
    Ratings = factor(Ratings, levels = c("Low", "High"), ordered = TRUE)
  )

# Verify the structure and summary of the prepared data
glimpse(selected_data)
summary(selected_data)

# Splitting data into training and testing sets
set.seed(123) # for reproducibility
train_index <- createDataPartition(selected_data$Ratings, p = 0.7, list = FALSE)
train_data <- selected_data[train_index, ]
test_data <- selected_data[-train_index, ]

cat("\nTraining Data Dimensions:", dim(train_data), "\n")
cat("Testing Data Dimensions:", dim(test_data), "\n")



# Analysis 3.1.1-1: What is the significance linear association between 
# independent variables like , Income levels, Total_Purchases, Product_Category and high Ratings?

# ----- Technique 1: Binary Logistic Regression -----

# Train Binary Logistic Regression model
logistic_model <- glm(Ratings ~ Income + Customer_Segment + Total_Purchases +
                        Product_Category + Shipping_Method + Payment_Method +
                        Order_Status,
                      data = train_data,
                      family = "binomial")

# Summary of the model
print(summary(logistic_model))

# Interpret coefficients (odds ratios)
print(exp(coef(logistic_model)))

# Predict probabilities on the test set
test_probabilities_logistic <- predict(logistic_model, newdata = test_data, type = "response")

# Convert probabilities to class predictions using a threshold like 0.5
test_predictions_logistic <- ifelse(test_probabilities_logistic > 0.5, "High", "Low")
test_predictions_logistic <- factor(test_predictions_logistic, levels = c("Low", "High"), ordered = TRUE)

# Use confusion matrix to evaluate model performance
conf_matrix_logistic <- confusionMatrix(test_predictions_logistic, test_data$Ratings)
print(conf_matrix_logistic)

# ROC Curve and AUC for Logistic Regression
pred_logistic <- prediction(test_probabilities_logistic, test_data$Ratings)
perf_logistic <- performance(pred_logistic, "tpr", "fpr")
auc_logistic <- performance(pred_logistic, "auc")@y.values[[1]]

cat("\nAUC for Logistic Regression:", auc_logistic, "\n")

plot(perf_logistic, col = "blue", main = "ROC Curve: Logistic Regression & Random Forest")
abline(a = 0, b = 1, lty = 2, col = "gray") # Add diagonal line



# Analysis 3.1.1-2: Based on the complex decision-making process of Random Forest model,
# which independent variables are identified as the most significant predictors for customer Ratings?
# ----- Technique 2: Random Forest for Binary Classification -----

# Train the Random Forest model using ranger
num_features <- ncol(train_data) - 1 # Exclude the response variable
mtry_val <- floor(sqrt(num_features))

rf_model <- ranger(Ratings ~ .,
                   data = train_data,
                   num.trees = 500,
                   mtry = mtry_val,
                   importance = "impurity",
                   probability = TRUE, # To get probabilities for ROC curve
                   seed = 123)

print(rf_model)

# Predict probabilities on the test set
test_predictions_rf_probs <- predict(rf_model, data = test_data, type = "response")$predictions

# We need the probability of the 'High' class for ROC
test_probabilities_rf <- test_predictions_rf_probs[, "High"]

# Convert probabilities to class predictions
test_predictions_rf_class <- ifelse(test_probabilities_rf > 0.5, "High", "Low")
test_predictions_rf_class <- factor(test_predictions_rf_class, levels = c("Low", "High"), ordered = TRUE)

# Evaluate model performance using a confusion matrix
conf_matrix_rf <- confusionMatrix(test_predictions_rf_class, test_data$Ratings)
print(conf_matrix_rf)

# ROC Curve and AUC for Random Forest
pred_rf <- prediction(test_probabilities_rf, test_data$Ratings)
perf_rf <- performance(pred_rf, "tpr", "fpr")
auc_rf <- performance(pred_rf, "auc")@y.values[[1]]

cat("\nAUC for Random Forest:", auc_rf, "\n")

# Add Random Forest ROC to the existing plot
plot(perf_rf, add = TRUE, col = "red")
legend("bottomright", legend = c(paste0("Logistic Regression (AUC = ", round(auc_logistic, 3), ")"),
                                 paste0("Random Forest (AUC = ", round(auc_rf, 3), ")")),
       col = c("blue", "red"), lty = 1)

# Variable Importance Plot
var_importance <- as.data.frame(sort(importance(rf_model), decreasing = TRUE))
colnames(var_importance) <- "Importance"
var_importance$Variable <- rownames(var_importance)

ggplot(var_importance, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Random Forest Variable Importance", x = "Variable", y = "Importance (Mean Decrease Impurity)") +
  theme_minimal()




# Analysis 3.1.1-3: How does the non-linear patterns in Total_Purchases affects 
# the probability of a 'High' rating, and how do these effects different categorical 
# variables like Product_Category or Income levels?

# ----- Technique 3: Partial Dependence Plots (PDPs) for Random Forest -----

# PDP for Total_Purchases (numeric variable)
pdp_total_purchases <- pdp::partial(rf_model,
                                    pred.var = "Total_Purchases",
                                    pred.grid = expand.grid(Total_Purchases = seq(min(selected_data$Total_Purchases),
                                                                                  max(selected_data$Total_Purchases),
                                                                                  length.out = 100)),
                                    plot = FALSE,
                                    chull = FALSE, # Set to FALSE for standard PDP
                                    progress = TRUE,
                                    which.class = "High") # Specify which class's probability to plot

autoplot(pdp_total_purchases, type = "lines", ylab = "Ratings = High") +
  geom_line(color = "darkgreen", size = 1.2) +
  labs(title = "Partial Dependence of Total_Purchases") +
  theme_minimal()

# PDP for a categorical variable (e.g., Income)
pdp_income <- pdp::partial(rf_model,
                           pred.var = "Income",
                           plot = FALSE,
                           chull = FALSE,
                           progress = TRUE,
                           which.class = "High")

autoplot(pdp_income, type = "bar", fill = "purple", alpha = 0.7, color = "black") +
  labs(title = "Partial Dependence of Income",
       y = "Ratings = High") +
  theme_minimal()

# PDP for interaction between a numeric and a categorical variable (e.g., Total_Purchases and Product_Category)
# Create a smaller grid for product_category to prevent too many lines/facets
unique_product_categories <- 
  levels(selected_data$Product_Category)[1:min(length(levels(selected_data$Product_Category)), 5)] # Plot up to 5 categories if too many

pdp_interaction <- pdp::partial(rf_model,
                                pred.var = c("Total_Purchases", "Product_Category"),
                                pred.grid = expand.grid(Total_Purchases = seq(min(selected_data$Total_Purchases),
                                                                              max(selected_data$Total_Purchases),
                                                                              length.out = 50),
                                                        Product_Category = unique_product_categories),
                                plot = FALSE,
                                chull = FALSE,
                                progress = TRUE,
                                which.class = "High")

# Plot the interaction with facets
ggplot(pdp_interaction, aes(x = Total_Purchases, y = yhat, color = Product_Category)) +
  geom_line(size = 1.2) +
  labs(title = "Total_Purchases & Product_Category",
       y = "Ratings = High",
       color = "Product Category") +
  theme_minimal() +
  facet_wrap(~Product_Category) # Separate plots for each category






# ------------------------------------------------------------------------------




# Objective 2: To examine the association between customers’ income levels and customers' ratings. 
# - Gavin Ma Wei Zhen TP080501

# Q1. What are the distribution of each income levels customers' ratings numbers?
# Count the proportion of each income levels customers' ratings.
# Both number count and proportion

# create a tibble for Percentage of Each Income Level
retail_ma %>%
  group_by(Income, Rating_Binary) %>%
  summarise(Number_Of_Ratings = n(),.groups = "keep") %>%
  group_by(Income) %>%
  mutate(Percentage_of_Each_Income_Level = Number_Of_Ratings / sum(Number_Of_Ratings) * 100) %>%
  arrange(Income) %>%
  filter(Rating_Binary == "1")

# Proportion visualization (stack bar plot)
ggplot(retail_ma, aes(x = Income, fill = as.factor(Rating_Binary))) +
  geom_bar(position = "fill") +
  # scale proportion to percentage format
  scale_y_continuous(labels = scales::percent_format()) + 
  labs(x = "Income Level", y = "Proportion", fill = "Ratings",
       title = "Proportion of ratings from customers") +
  scale_fill_manual(values = c("0" = "red", "1" = "steelblue"),
                    labels = c("Low", "High")) +
  theme_minimal()


# Count visualization (Clustered Bar Plot)
retail_ma %>%
  group_by(Income, Rating_Binary) %>%
  summarise(Number_Of_Ratings= n())

ggplot(retail_ma, aes(x = Income, fill = Income)) +
  geom_bar(position = "dodge") +
  # use after_stat(count) at ggplot2 future versions, 
  # dot-dot notation (`..count..`) was deprecated in ggplot2 3.4.0.
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.4, size = 3.5) +
  # scale count from scientific notation to numbers with comma
  scale_y_continuous(labels = comma) + 
  labs(x = "Ratings", y = "Count", fill = "Income Level",
       title = "Number of customers ratings by income level") +
  theme_minimal() +
  scale_fill_manual(values = c("Low" = "red", "Medium" = "steelblue", "High" = "blue"),
                    labels = c("Low", "Medium", "High")) +
  facet_wrap(~Ratings, scale = "free")

# Cross Table of Ratings by Income Levels
Income_table = table(retail_ma$Income, retail_ma$Rating_Binary)
colnames(Income_table) <- c("Low Rating", "High Rating") # change column name
print(Income_table)

# Calculate proportions
prop.table(Income_table, margin = 1) * 100

# Q2. How is the customer's income level associated with ratings?
# Test the association between Income Level and Ratings.

# Chi-square test of independence

# Set hypothesis
# H0: no association between Customer Income Level and Ratings
# H1: have association between Customer Income Level and Ratings

# create table 
table = table(retail_ma$Income, retail_ma$Rating_Binary)
table

# chi-square test
chisq <- chisq.test(table)
chisq

# Check expected frequencies (ensure validity)
chisq$expected

# p-value < 2.2e-16 (2.2*10^(-16), 0.00000000000000022)
# 2.2e-16 (p-value) < 0.05(significance level)
# Assume significance level of 0.05 or 5%,
# Since the p-value is smaller than the significance level(0.05), 
# reject the null hypothesis. 
# There are association between Income Level and Ratings.

# Calculate Cramer's V
cramer_v <- sqrt(chisq$statistic / (nrow(retail_ma) * min(dim(table) - 1)))
print(paste("Cramer's V:", round(cramer_v, 5)))

# Q3. 
# Logistic regression: predict Rating_Binary using Income and other segments

# Change reference term
retail_ma$Income <- relevel(retail_ma$Income, ref = "Low")

set.seed(123)
split = sample.split(retail_ma$Rating_Binary, SplitRatio = 0.8)
# training set is to train the model to learn about the data
training_set = subset(retail_ma, split == TRUE)
# test set is to use the trained model to predict the outcome
test_set = subset(retail_ma, split == FALSE)

# build model
model <- glm(Rating_Binary ~ Income + Customer_Segment + Payment_Method + Order_Status,
             data = training_set, family = binomial)
model
# model summary
summary(model)

# broom package use to convert coefficients to odds ratios
# to make interpretation more easier
library(broom)

library(car)  # For VIF calculation
library(ggeffects)  # For marginal effects

# Get odds ratios
exp(coef(model))
tidy(model, exponentiate = TRUE, conf.int = TRUE)

# estimate of giving high rating
# Low 2.5142231
# Medium 2.1001498
# High 1.9007292

# predict test set
# Predicted probabilities
pred_probs <- predict(model, type = "response", test_set[,-8])
pred_probs
# Classify based on 0.5 threshold
pred_class <- ifelse(pred_probs > 0.5, 1, 0)
pred_class


# Confusion matrix 
confusionMatrix(table(Predicted = pred_class, Actual = test_set$Rating_Binary))

# Probability table for training_set's Rating_Binary
prop.table(table(training_set$Rating_Binary))
prop.table(table(test_set$Rating_Binary))


# Visualize logistic regression model with ROC Curve
library(pROC)

roc_obj <- roc(test_set$Rating_Binary, pred_probs)

# Basic plot
plot(roc_obj, col = "blue", main = "ROC Curve - Logistic Regression")

# Add AUC to plot
auc_val <- auc(roc_obj)
legend("bottomright", legend = paste("AUC =", round(auc_val, 3)), col = "blue", lwd = 2)


# Q4. How does Income compare to other features (Customer_Segment, Total_Purchases, 
#     Order_Status, etc.) in predicting Customer Satisfaction?

#  Calculate proportions of High ratings for each combination between Income and Customer_Segment
# Prepare proportion data
proportion_data <- retail_ma %>%
  group_by(Income, Customer_Segment, Rating_Binary) %>%
  summarise(count = n(),.groups = "keep") %>%
  group_by(Income, Customer_Segment) %>%
  mutate(proportion = count / sum(count)) %>%
  mutate(percentage = proportion * 100) %>%
  filter(Rating_Binary == 1)
proportion_data

# Clustered bar plot
ggplot(proportion_data, aes(x = Income, y = proportion, fill = Customer_Segment)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Proportion of High Ratings by Income and Customer Segment",
       x = "Income Level",
       y = "Proportion of High Ratings",
       fill = "Customer Segment") +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_manual(values = c("Regular" = "purple", "Premium" = "yellow", "New" = "red")) +
  theme_minimal() +
  theme(legend.position = "right")




# ------------------------------------------------------------------------------





View(retail_ma)

# Objective 3: To assess the relationship between customer segments and customers’ ratings. 
# - Tey Kai Yuan TP081603

# Q1: Do customer segments have significantly different satisfaction levels?
# Create a contingency table (cross-tabulation) of Customer Segment vs. Ratings
rating_table <- table(retail$Customer_Segment, retail$Ratings)
table(retail$Customer_Segment, retail$Ratings)

# Perform Chi-square test to check if Customer Segment and Ratings are independent
chisq.test(rating_table)

# Plot a stacked bar chart showing the proportion of each rating within customer segments
ggplot(retail, aes(x = Customer_Segment, fill = factor(Ratings))) +
  geom_bar(position = "fill") +  # position = "fill" scales bars to 100% height (proportion)
  scale_y_continuous(labels = scales::percent_format()) +  # Format y-axis labels as percentages
  labs(title = "Ratings Distribution by Customer Segment",
       x = "Customer Segment", y = "Proportion", fill = "Rating (0 = Low, 1 = High)") +
  theme_minimal()


# Q2: What proportion of high ratings does each customer segment give?
# Group data by Customer Segment and calculate the mean of Ratings (binary 0/1),
# which gives the proportion of high ratings in each segment
segment_rating <- retail %>%
  group_by(Customer_Segment) %>%
  summarise(High_Rating_Proportion = mean(Ratings, na.rm = TRUE))

# Plot a bar chart showing the proportion of high ratings by customer segment
ggplot(segment_rating, aes(x = reorder(Customer_Segment, -High_Rating_Proportion),
                           y = High_Rating_Proportion, fill = Customer_Segment)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Proportion of High Ratings by Customer Segment",
       x = "Customer Segment", y = "High Rating Proportion") +
  theme_minimal()


# Q3: Which combination of customer segment and payment method leads to the highest satisfaction?
# Group by Customer Segment and Payment Method and calculate the proportion of high ratings
combo_rating <- retail %>%
  group_by(Customer_Segment, Payment_Method) %>%
  summarise(High_Proportion = mean(Ratings, na.rm = TRUE))

# Plot a heatmap showing high rating proportion for each customer segment and payment method combination
ggplot(combo_rating, aes(x = Payment_Method, y = Customer_Segment, fill = High_Proportion)) +
  geom_tile(color = "white") +  # Tiles with white borders
  scale_fill_gradient(low = "moccasin", high = "navajowhite4", labels = scales::percent_format()) +
  labs(title = "High Rating Proportion by Segment and Payment Method",
       x = "Payment Method", y = "Customer Segment", fill = "High Rating %") +
  theme_minimal()


# Q4: How does customer satisfaction vary across segments and product categories?
# Ensure Ratings is numeric (0/1)
retail$Ratings <- as.numeric(retail$Ratings)

# Group data by Customer Segment and Product Category,
# calculate the proportion of high ratings within each group
combo_segment_product <- retail %>%
  group_by(Customer_Segment, Product_Category) %>%
  summarise(High_Rating_Proportion = mean(Ratings, na.rm = TRUE))

# Plot faceted bar chart showing high rating proportions by product category,
# with a separate facet for each customer segment
ggplot(combo_segment_product, aes(x = Product_Category, y = High_Rating_Proportion, fill = Product_Category)) +
  geom_col() +
  facet_wrap(~ Customer_Segment) +
  labs(title = "High Rating Proportion by Segment and Product Category",
       x = "Product Category", y = "High Rating Proportion") +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for readability

# Q5: Can customers be clustered based on their ratings, purchases, and income?
# Select the relevant features for clustering: Ratings, Total Purchases, and Income
cluster_data <- retail %>%
  dplyr::select(Ratings, Total_Purchases, Income)

# Check for any missing or infinite values in the selected columns
cat("Missing values in selected columns:\n")
print(colSums(is.na(cluster_data)))
cat("Infinite values in selected columns:\n")
print(colSums(sapply(cluster_data, is.infinite)))

# Remove rows with missing or infinite values to ensure clean data
cluster_data_clean <- cluster_data[complete.cases(cluster_data), ]

# Validate that there is a sufficient amount of data for clustering
min_rows_required <- 3
if(nrow(cluster_data_clean) < min_rows_required){
  stop("Not enough valid data rows for clustering. Need at least ", min_rows_required)
}

# Normalize the features to ensure all variables contribute equally to clustering
cluster_data_scaled <- scale(cluster_data_clean)

# Determine the optimal number of clusters using the Elbow Method (on a sample to save memory)
k_default <- 3
k_max <- 5
sample_size <- 3000  # Limits the sample size to improve efficiency

if (nrow(cluster_data_scaled) >= k_max) {
  tryCatch({
    sample_index <- sample(1:nrow(cluster_data_scaled), min(sample_size, nrow(cluster_data_scaled)))
    elbow_sample <- cluster_data_scaled[sample_index, ]
    
    # Visualize within-cluster sum of squares for different values of k
    elbow_plot <- fviz_nbclust(elbow_sample, kmeans, method = "wss", k.max = k_max, nstart = 10) +
      labs(title = "Elbow Method for Optimal Number of Clusters (Sampled)") +
      theme_minimal()
    print(elbow_plot)
  }, error = function(e) {
    # If the Elbow plot fails, fallback to default cluster count
    cat("\nError in fviz_nbclust: ", e$message, "\nUsing default k =", k_default, "\n")
  })
} else {
  # If sample size is too small, use default cluster count
  cat("\nNot enough rows for Elbow plot. Using default k =", k_default, "\n")
}

# Apply K-means clustering to assign customers into groups
set.seed(123)  # For reproducibility
kmeans_result <- kmeans(cluster_data_scaled, centers = k_default, nstart = 10)

# Append the cluster assignments back to the original cleaned dataset
clustered_data <- cluster_data_clean
clustered_data$Cluster <- as.factor(kmeans_result$cluster)

# Print the number of customers in each cluster
cat("\nCluster Summary:\n")
print(table(clustered_data$Cluster))

# Create a summary of the proportion of high ratings in each cluster
rating_summary <- clustered_data %>%
  group_by(Cluster) %>%
  summarise(High_Rating_Proportion = mean(Ratings, na.rm = TRUE))

# Visualize the average rating proportion per cluster
ggplot(rating_summary, aes(x = Cluster, y = High_Rating_Proportion, fill = Cluster)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Proportion of High Ratings by Cluster",
       x = "Cluster", y = "High Rating Proportion") +
  theme_minimal()

# Scatter plot to visualize how clusters differ by Income and Purchases, shaped by Ratings
ggplot(clustered_data, aes(x = Total_Purchases, y = Income, color = Cluster)) +
  geom_point(aes(shape = factor(Ratings)), size = 2, alpha = 0.7) +
  labs(title = "Customer Clusters by Purchases, Income, and Ratings",
       x = "Total Purchases（Million）", y = "Income", shape = "Rating", color = "Cluster") +
  theme_minimal()




# ------------------------------------------------------------------------------





# Objective 4: To investigate the impact of the total amount of products purchased by 
# customers against customer.rating. – Leok Chun Bin TP080536

# Question 1
# Can we predict customer satisfaction using Total_Purchases?
library(randomForest)
library(caret)
library(ggplot2)

retail$Income = as.factor(retail$Income)
retail$Customer_Segment = as.factor(retail$Customer_Segment)
retail$Ratings = as.factor(retail$Ratings)

table(retail$Ratings)

set.seed(123)
rf_model_balanced = randomForest(Ratings ~ Total_Purchases + Income + Customer_Segment,
                                 data = retail,
                                 ntree = 200,
                                 importance = TRUE,
                                 strata = retail$Ratings,
                                 sampsize = c(100000, 100000))  

retail$rf_pred_balanced = predict(rf_model_balanced)

conf_matrix = confusionMatrix(retail$rf_pred_balanced, retail$Ratings)
print(conf_matrix)

ggplot(retail, aes(x = rf_pred_balanced, fill = Ratings)) +
  geom_bar(position = "dodge") +
  labs(title = "Random Forest (Balanced): Predicted vs Actual Customer Ratings",
       x = "Predicted Rating", fill = "Actual Rating")

varImpPlot(rf_model_balanced, main = "Random Forest Feature Importance (Balanced)")


# Question 2
# How does customer satisfaction vary across different purchase volume segments?
library(dplyr)
library(ggplot2)

retail$Purchase_Level = cut(retail$Total_Purchases,
                            breaks = quantile(retail$Total_Purchases, probs = c(0, 0.33, 0.66, 1), na.rm = TRUE),
                            labels = c("Low", "Medium", "High"),
                            include.lowest = TRUE)

purchase_rating_table = table(retail$Purchase_Level, retail$Ratings)
print(purchase_rating_table)

chisq_result = chisq.test(purchase_rating_table)
print(chisq_result)

mosaicplot(purchase_rating_table,
           main = "Satisfaction by Purchase Volume Segment",
           color = TRUE, shade = TRUE, xlab = "Purchase Level", 
           ylab = "Satisfaction Rating")



# Question 3
# How important is Total_Purchases compared to other features in predicting customer satisfaction?
library(dplyr)
library(ggplot2)

retail$Ratings = as.factor(retail$Ratings)
retail$Income = as.factor(retail$Income)
retail$Customer_Segment = as.factor(retail$Customer_Segment)
retail$Shipping_Method = as.factor(retail$Shipping_Method)

group_rare_levels = function(x, threshold = 5000) {
  freq = table(x)
  x = as.character(x)
  x[!(x %in% names(freq[freq >= threshold]))] = "Other"
  return(as.factor(x))
}

retail$Product_Type = group_rare_levels(retail$Product_Type)
retail$Product_Category = group_rare_levels(retail$Product_Category)

model_logit = glm(Ratings ~ Total_Purchases + Income + Customer_Segment +
                    Shipping_Method + Product_Type + Product_Category,
                  data = retail,
                  family = binomial())

summary(model_logit)

retail$Purchase_Bin = cut(retail$Total_Purchases,
                          breaks = quantile(retail$Total_Purchases, probs = 
                                              c(0, 0.33, 0.66, 1), na.rm = TRUE),
                          labels = c("Low", "Medium", "High"),
                          include.lowest = TRUE)

satisfaction_summary = retail %>%
  group_by(Purchase_Bin) %>%
  summarise(SatisfactionRate = mean(as.numeric(as.character(Ratings))))  

ggplot(satisfaction_summary, aes(x = Purchase_Bin, y = SatisfactionRate)) +
  geom_col(fill = "steelblue") +
  ylim(0, 1) +
  labs(title = "Average Satisfaction Rate by Purchase Volume",
       x = "Total Purchase Level",
       y = "Satisfaction Rate") +
  theme_minimal()

# ------------------------------------------------------------------------------
