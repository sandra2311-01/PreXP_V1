from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

class MissingDataHandler:
    def __init__(self, insight_logs, logger, low_threshold=0.05, high_threshold=0.30, min_rows_percentage=0.90):
        """
        Initialize the MissingDataHandler using insight logs.

        Parameters:
            insight_logs (list): Logs containing insights about the dataset.
            logger (ReasoningLogger): Logger instance for preprocessing actions.
            low_threshold (float): Threshold for MCAR (default is 5%).
            high_threshold (float): Threshold for MNAR (default is 30%).
            min_rows_percentage (float): Minimum percentage of rows required after deletion (default is 90%).
        """
        self.insight_logs = insight_logs
        self.logger = logger
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.min_rows_percentage = min_rows_percentage

    def retrieve_missing_percentage(self, column):
        """
        Retrieve the missing percentage for a specific column from insight logs.

        Parameters:
            column (str): The column name.

        Returns:
            float: Missing percentage for the column.
        """
        for log in self.insight_logs:
            if log['column'] == column and 'p_missing' in log['metrics']:
                return log['metrics']['p_missing']
        return 0.0

    def categorize_missing_data(self, df):
        """
        Categorize columns with missing data into MCAR, MAR, or MNAR.

        Parameters:
            df (pd.DataFrame): The DataFrame to categorize missing data.

        Returns:
            dict: Categorized columns as 'MCAR', 'MAR', and 'MNAR'.
        """
        categorized_columns = {'MCAR': [], 'MAR': [], 'MNAR': []}
        for column in df.columns:
            missing_percentage = self.retrieve_missing_percentage(column)

            if missing_percentage < self.low_threshold:
                categorized_columns['MCAR'].append(column)
                self.logger.log_preprocess(
                    step="Missing Data Categorization",
                    column=column,
                    details=f"Categorized as MCAR because missing percentage ({missing_percentage}%) is below the low threshold ({self.low_threshold}%).",
                    metrics={"missing_percentage": missing_percentage},
                    decision="MCAR"
                )
            elif self.low_threshold <= missing_percentage <= self.high_threshold:
                categorized_columns['MAR'].append(column)
                self.logger.log_preprocess(
                    step="Missing Data Categorization",
                    column=column,
                    details=f"Categorized as MAR because missing percentage ({missing_percentage}%) is between {self.low_threshold}% and {self.high_threshold}%.",
                    metrics={"missing_percentage": missing_percentage},
                    decision="MAR"
                )
            else:
                categorized_columns['MNAR'].append(column)
                self.logger.log_preprocess(
                    step="Missing Data Categorization",
                    column=column,
                    details=f"Categorized as MNAR because missing percentage ({missing_percentage}%) exceeds the high threshold ({self.high_threshold}%).",
                    metrics={"missing_percentage": missing_percentage},
                    decision="MNAR"
                )

        return categorized_columns

    def handle_missing_data(self, df, missing_data_categories):
        """
        Handle missing data based on its categorization.

        Parameters:
            df (pd.DataFrame): DataFrame with missing data.
            missing_data_categories (dict): Categorized columns as 'MCAR', 'MAR', and 'MNAR'.

        Returns:
            pd.DataFrame: The DataFrame with missing data handled.
        """
        initial_row_count = len(df)
        min_rows = int(initial_row_count * self.min_rows_percentage)

        self.logger.log_preprocess(
            step="Missing Data Handling",
            column=None,
            details="Initial row count and minimum rows calculated for handling missing data.",
            metrics={"initial_row_count": initial_row_count, "min_rows_required": min_rows}
        )

        # Handle MCAR
        for column in missing_data_categories['MCAR']:
            missing_count = df[column].isna().sum()
            rows_with_non_missing = initial_row_count - missing_count

            if rows_with_non_missing >= min_rows:
                df = df.dropna(subset=[column])
                self.logger.log_preprocess(
                    step="MCAR Handling",
                    column=column,
                    details=f"Dropped rows with missing values since non-missing rows ({rows_with_non_missing}) met the minimum required ({min_rows}).",
                    decision="Drop Rows"
                )
            else:
                strategy = 'most_frequent' if df[column].dtype == 'object' else 'mean'
                imputer = SimpleImputer(strategy=strategy)
                df[column] = imputer.fit_transform(df[[column]])
                self.logger.log_preprocess(
                    step="MCAR Handling",
                    column=column,
                    details=f"Applied imputation using '{strategy}' because dropping rows would violate minimum row requirement.",
                    decision=f"Impute with {strategy}"
                )

        # Handle MAR
        for column in missing_data_categories['MAR']:
            missing_count = df[column].isna().sum()
            if df[column].dtype != 'object':
                imputer = IterativeImputer(random_state=42)
                df[[column]] = imputer.fit_transform(df[[column]])
                self.logger.log_preprocess(
                    step="MAR Handling",
                    column=column,
                    details="Applied iterative imputation because the column is numeric and missing percentage is moderate.",
                    metrics={"missing_count": missing_count},
                    decision="Advanced Imputation"
                )
            else:
                strategy = 'most_frequent'
                imputer = SimpleImputer(strategy=strategy)
                df[column] = imputer.fit_transform(df[[column]])
                self.logger.log_preprocess(
                    step="MAR Handling",
                    column=column,
                    details="Applied simple imputation with 'most_frequent' because the column is categorical.",
                    decision="Simple Imputation"
                )

        # Handle MNAR
        for column in missing_data_categories['MNAR']:
            missing_count = df[column].isna().sum()
            df[column + '_missing'] = df[column].isnull().astype(int)
            strategy = 'most_frequent' if df[column].dtype == 'object' else 'median'
            imputer = SimpleImputer(strategy=strategy)
            df[column] = imputer.fit_transform(df[[column]])
            self.logger.log_preprocess(
                step="MNAR Handling",
                column=column,
                details=f"Added missing indicator column and applied imputation using '{strategy}' to handle MNAR.",
                metrics={"missing_count": missing_count},
                decision=f"Indicator + Impute with {strategy}"
            )

        return df
