from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from scipy.stats import kendalltau
import pandas as pd


class DataEncoder:
    def __init__(self, df, insights, logger=None):
        """
        Initialize the DataEncoder with a DataFrame, profiling insights, and optional logger.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            insights (dict): Profiling insights providing column information.
            logger (ReasoningLogger, optional): Logger to store preprocessing logs.
        """
        self.df = df
        self.insights = insights
        self.logger = logger

    def encode_columns(self, target_column=None, low_cardinality_threshold=3, high_cardinality_threshold=20):
        """
        Encodes categorical, boolean, numeric, and datetime columns based on insights.

        Parameters:
            target_column (str, optional): Name of the target column for ordinal detection.
            low_cardinality_threshold (int): Threshold for applying One-Hot Encoding.
            high_cardinality_threshold (int): Threshold for applying Label Encoding.

        Returns:
            pd.DataFrame: DataFrame with encoded columns.
        """
        df_encoded = self.df.copy()
        processed_columns = set()  # Track processed columns

        # Encode Categorical Columns
        for column, details in self.insights.get('variables', {}).items():
            if column in processed_columns or column == target_column:
                continue

            column_type = details.get('type')
            num_unique_values = details.get('n_distinct', df_encoded[column].nunique())

            if column_type not in ["Categorical", "Mixed", "Text"]:
                continue

            # Handle missing values
            df_encoded[column] = df_encoded[column].fillna('Missing')

            # Check ordinal relationship with target column
            if target_column and pd.api.types.is_numeric_dtype(df_encoded[target_column]):
                df_subset = df_encoded[[column, target_column]].dropna()
                df_subset[column] = df_subset[column].astype('category').cat.codes
                tau, _ = kendalltau(df_subset[column], df_subset[target_column])
                if abs(tau) > 0.5:
                    if self.logger:
                        self.logger.log_preprocess(
                            step="Encoding",
                            column=column,
                            details=f"Detected as ordinal based on high Kendall's Tau correlation (Tau: {tau:.2f}).",
                            metrics={"method": "Ordinal Encoding", "Kendall's Tau": tau},
                            decision="Ordinal Encoding, because Ordinal relationship with target."
                        )
                    le = LabelEncoder()
                    df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
                    processed_columns.add(column)
                    continue

            # Choose encoding method based on cardinality
            if num_unique_values <= low_cardinality_threshold:
                if self.logger:
                    self.logger.log_preprocess(
                        step="Encoding",
                        column=column,
                        details=f"Low cardinality ({num_unique_values} unique values).",
                        metrics={"method": "One-Hot Encoding"},
                        decision="One-Hot Encoding, because it simplifies categorical variable."
                    )
                df_encoded = pd.get_dummies(df_encoded, columns=[column], drop_first=True)
            elif num_unique_values <= high_cardinality_threshold:
                if self.logger:
                    self.logger.log_preprocess(
                        step="Encoding",
                        column=column,
                        details=f"Moderate cardinality ({num_unique_values} unique values).",
                        metrics={"method": "Label Encoding"},
                        decision="Label Encoding, because efficient for moderate cardinality."
                    )
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
            else:
                if self.logger:
                    self.logger.log_preprocess(
                        step="Encoding",
                        column=column,
                        details=f"High cardinality ({num_unique_values} unique values).",
                        metrics={"method": "Frequency Encoding"},
                        decision="Frequency Encoding, reduces dimensionality efficiently."
                    )
                frequencies = df_encoded[column].value_counts(normalize=True)
                df_encoded[column] = df_encoded[column].map(frequencies)

            processed_columns.add(column)  # Mark column as processed

        # Encode Boolean Columns
        boolean_columns = df_encoded.select_dtypes(include=['bool']).columns
        for column in boolean_columns:
            if column in processed_columns:
                continue
            if self.logger:
                self.logger.log_preprocess(
                    step="Encoding",
                    column=column,
                    details="Boolean column detected.",
                    metrics={"method": "Converted to 1/0"},
                    decision="Converted to 1/0, it standardize boolean format."
                )
            df_encoded[column] = df_encoded[column].astype(int)
            processed_columns.add(column)

        # Handle Numeric Columns
        numeric_columns = df_encoded.select_dtypes(include=['int64', 'float64']).columns
        for column in numeric_columns:
            if column in processed_columns or column == target_column:
                continue
            if self.logger:
                self.logger.log_preprocess(
                    step="Encoding",
                    column=column,
                    details="Numeric column detected.",
                    metrics={"method": "Retained as is"},
                    decision="No transformation applied."
                )
            processed_columns.add(column)

        # Handle Date Columns
        date_columns = df_encoded.select_dtypes(include=['datetime64']).columns
        for column in date_columns:
            if column in processed_columns:
                continue
            if self.logger:
                self.logger.log_preprocess(
                    step="Encoding",
                    column=column,
                    details="Datetime column detected.",
                    metrics={"method": "Extracted date components"},
                    decision="Extracted date components, Decomposed into year, month, day, weekday."
                )
            df_encoded[f"{column}_year"] = df_encoded[column].dt.year
            df_encoded[f"{column}_month"] = df_encoded[column].dt.month
            df_encoded[f"{column}_day"] = df_encoded[column].dt.day
            df_encoded[f"{column}_weekday"] = df_encoded[column].dt.weekday
            df_encoded.drop(columns=[column], inplace=True)
            processed_columns.add(column)

        return df_encoded
