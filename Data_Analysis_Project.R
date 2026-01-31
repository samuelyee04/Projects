
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






