from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
import pandas as pd

#class ScalingAndOutlierHandler:
#    def __init__(self, df, insights, logger):
#        """
#        Initialize the Scaling and Outlier Handler with the DataFrame, insights, and logger.
#
#        Parameters:
#            df (pd.DataFrame): The input DataFrame.
#            insights (dict): Extracted insights containing column details.
#            logger (ReasoningLogger): Logger for recording preprocessing logs.
#        """
#        self.df = df
#        self.insights = insights
#        self.logger = logger
#
#    def determine_scaling_and_outlier_handling(self, drop_outliers=True, exclude_outliers=None, percentile_threshold=90):
#        """
#        Determine scaling methods and outlier handling decisions based on insights.
#
#        Parameters:
#            drop_outliers (bool): Whether to drop outliers by default.
#            exclude_outliers (list): Columns for which outliers should not be dropped.
#            percentile_threshold (int): Percentile for dynamic range threshold.
#
#        Returns:
#            tuple: Scaling and outlier handling decisions.
#        """
#        exclude_outliers = exclude_outliers or []
#        ranges = []
#
#        # Extract range information from insights
#        for column, details in self.insights['variables'].items():
#            if details['type'] in ['Numeric', 'Integer']:
#                col_min = details.get('min', 0)
#                col_max = details.get('max', 0)
#                ranges.append(col_max - col_min)
#
#        # Compute dynamic range threshold
#        range_threshold = np.percentile(ranges, percentile_threshold) if ranges else 0
#
#        scaling_decisions = {}
#        outlier_decisions = {}
#
#        for column, details in self.insights['variables'].items():
#            if details['type'] in ['Numeric', 'Integer']:
#                skewness = details.get('skewness', 0)
#                col_min, col_max = details.get('min', 0), details.get('max', 0)
#                range_value = col_max - col_min
#                n = details.get('n', 1)  # Avoid division by zero
#                n_outliers = details.get('n_outliers', 0)
#                outlier_percentage = (n_outliers / n) * 100 if n > 0 else 0
#
#                # Determine scaling method
#                if abs(skewness) > 2 or range_value > range_threshold:
#                    scaling_decisions[column] = 'normalization'
#                elif outlier_percentage > 5:
#                    scaling_decisions[column] = 'robust_scaling'
#                else:
#                    scaling_decisions[column] = 'standardization'
#
#                # Determine outlier handling
#                if outlier_percentage == 0:
#                    outlier_decisions[column] = 'no_outliers'
#                elif column in exclude_outliers:
#                    outlier_decisions[column] = 'include'
#                elif drop_outliers and outlier_percentage > 5:
#                    outlier_decisions[column] = 'drop'
#                else:
#                    outlier_decisions[column] = 'include'
#
#                # Log preprocessing decisions
#                self.logger.log_preprocess(
#                    step="Scaling and Outlier Handling",
#                    column=column,
#                    details="Determined scaling and outlier handling decisions.",
#                    metrics={
#                        "Skewness": skewness,
#                        "Range": range_value,
#                        "Outlier Percentage": outlier_percentage,
#                        "Scaling Method": scaling_decisions[column],
#                        "Outlier Handling": outlier_decisions[column]
#                    },
#                    decision=f"Scaling: {scaling_decisions[column]}, Outlier Handling: {outlier_decisions[column]}"
#                )
#
#        return scaling_decisions, outlier_decisions
#
#    def apply_scaling_and_outliers(self, scaling_decisions, outlier_decisions):
#        """
#        Apply scaling and handle outliers based on decisions.
#
#        Parameters:
#            scaling_decisions (dict): Scaling decisions for each column.
#            outlier_decisions (dict): Outlier handling decisions for each column.
#
#        Returns:
#            pd.DataFrame: Scaled and cleaned DataFrame.
#        """
#        df_processed = self.df.copy()
#
#        for column, method in scaling_decisions.items():
#            if column in outlier_decisions:
#                if outlier_decisions[column] == 'drop':
#                    # Remove outliers using IQR
#                    q1 = df_processed[column].quantile(0.25)
#                    q3 = df_processed[column].quantile(0.75)
#                    iqr = q3 - q1
#                    lower_bound = q1 - 1.5 * iqr
#                    upper_bound = q3 + 1.5 * iqr
#
#                    df_processed = df_processed[(df_processed[column] >= lower_bound) & (df_processed[column] <= upper_bound)]
#
#                    self.logger.log_preprocess(
#                        step="Outlier Handling",
#                        column=column,
#                        details="Dropped outliers using IQR.",
#                        metrics={"Lower Bound": lower_bound, "Upper Bound": upper_bound},
#                        decision="Outliers dropped"
#                    )
#
#                elif outlier_decisions[column] == 'include':
#                    self.logger.log_preprocess(
#                        column=column,
#                        details="Retained outliers as per decision.",
#                        metrics={},
#                        decision="Outliers retained"
#                    )
#
#            # Apply scaling to the column
#            scaler = None
#            if method == 'normalization':
#                scaler = MinMaxScaler()
#            elif method == 'standardization':
#                scaler = StandardScaler()
#            elif method == 'robust_scaling':
#                scaler = RobustScaler()
#
#            if scaler:
#                df_processed[column] = scaler.fit_transform(df_processed[[column]])
#                self.logger.log_preprocess(
#                    step="Scaling",
#                    column=column,
#                    details=f"Applied {method.capitalize()}.",
#                    metrics={},
#                    decision=f"Scaling applied: {method}"
#                )
#
#        return df_processed
#---------------------------------------------------------
#from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
#import numpy as np
#import pandas as pd

class ScalingAndOutlierHandler:
    def __init__(self, df, insights, logger):
        """
        Initialize the Scaling and Outlier Handler with the DataFrame, insights, and logger.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            insights (dict): Extracted insights containing column details.
            logger (ReasoningLogger): Logger for recording preprocessing logs.
        """
        self.df = df
        self.insights = insights
        self.logger = logger

    def determine_scaling_and_outlier_handling(self, drop_outliers=True, exclude_outliers=None, percentile_threshold=90):
        """
        Determine scaling methods and outlier handling decisions based on insights.

        Parameters:
            drop_outliers (bool): Whether to drop outliers by default.
            exclude_outliers (list): Columns for which outliers should not be dropped.
            percentile_threshold (int): Percentile for dynamic range threshold.

        Returns:
            tuple: Scaling and outlier handling decisions.
        """
        exclude_outliers = exclude_outliers or []
        ranges = []

        # Extract range information from insights
        for column, details in self.insights['variables'].items():
            if details['type'] in ['Numeric', 'Integer']:
                col_min = details.get('min', 0)
                col_max = details.get('max', 0)
                ranges.append(col_max - col_min)

        # Compute dynamic range threshold
        range_threshold = np.percentile(ranges, percentile_threshold) if ranges else 0

        scaling_decisions = {}
        outlier_decisions = {}

        for column, details in self.insights['variables'].items():
            if details['type'] in ['Numeric', 'Integer']:
                skewness = details.get('skewness', 0)
                col_min, col_max = details.get('min', 0), details.get('max', 0)
                range_value = col_max - col_min
                n = details.get('n', 1)  # Avoid division by zero
                n_outliers = details.get('n_outliers', 0)
                outlier_percentage = (n_outliers / n) * 100 if n > 0 else 0

                # Determine scaling method
                if abs(skewness) > 2:
                    scaling_decisions[column] = 'normalization'
                    scaling_reason = "High skewness detected; normalization scales to [0, 1] without affecting distribution."
                elif range_value > range_threshold:
                    scaling_decisions[column] = 'normalization'
                    scaling_reason = f"Range exceeds dynamic threshold ({range_threshold}). Normalization ensures all values fall within [0, 1]."
                elif outlier_percentage > 5:
                    scaling_decisions[column] = 'robust_scaling'
                    scaling_reason = "Significant outliers detected; robust scaling reduces their impact."
                else:
                    scaling_decisions[column] = 'standardization'
                    scaling_reason = "Low skewness and acceptable range; standardization centers data with unit variance."

                # Determine outlier handling
                if outlier_percentage == 0:
                    outlier_decisions[column] = 'no_outliers'
                    outlier_reason = "No outliers detected; no handling required."
                elif column in exclude_outliers:
                    outlier_decisions[column] = 'include'
                    outlier_reason = "Excluded from outlier handling as per user configuration."
                elif drop_outliers and outlier_percentage > 5:
                    outlier_decisions[column] = 'drop'
                    outlier_reason = "High outlier percentage; rows dropped using IQR method."
                else:
                    outlier_decisions[column] = 'include'
                    outlier_reason = "Outlier percentage low; retaining for analysis."

                # Log preprocessing decisions
                self.logger.log_preprocess(
                    step="Scaling and Outlier Handling",
                    column=column,
                    details=f"Scaling: {scaling_reason} Outlier Handling: {outlier_reason}",
                    metrics={
                        "Skewness": skewness,
                        "Range": range_value,
                        "Outlier Percentage": outlier_percentage,
                        "Scaling Method": scaling_decisions[column],
                        "Outlier Handling": outlier_decisions[column]
                    },
                    decision=f"Scaling: {scaling_decisions[column]}, Outlier Handling: {outlier_decisions[column]}"
                )

        return scaling_decisions, outlier_decisions

    def apply_scaling_and_outliers(self, scaling_decisions, outlier_decisions):
        """
        Apply scaling and handle outliers based on decisions.

        Parameters:
            scaling_decisions (dict): Scaling decisions for each column.
            outlier_decisions (dict): Outlier handling decisions for each column.

        Returns:
            pd.DataFrame: Scaled and cleaned DataFrame.
        """
        df_processed = self.df.copy()

        for column, method in scaling_decisions.items():
            # Outlier handling
            if column in outlier_decisions:
                if outlier_decisions[column] == 'drop':
                    q1 = df_processed[column].quantile(0.25)
                    q3 = df_processed[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    df_processed = df_processed[(df_processed[column] >= lower_bound) & (df_processed[column] <= upper_bound)]

                    self.logger.log_preprocess(
                        step="Outlier Handling",
                        column=column,
                        details=f"Dropped outliers using IQR method (Lower Bound: {lower_bound}, Upper Bound: {upper_bound}).",
                        metrics={"Lower Bound": lower_bound, "Upper Bound": upper_bound},
                        decision="Outliers dropped"
                    )

                elif outlier_decisions[column] == 'include':
                    self.logger.log_preprocess(
                        step="Outlier Handling",
                        column=column,
                        details="Retained outliers as per decision.",
                        metrics={},
                        decision="Outliers retained"
                    )

            # Scaling
            scaler = None
            if method == 'normalization':
                scaler = MinMaxScaler()
            elif method == 'standardization':
                scaler = StandardScaler()
            elif method == 'robust_scaling':
                scaler = RobustScaler()

            if scaler:
                df_processed[column] = scaler.fit_transform(df_processed[[column]])
                self.logger.log_preprocess(
                    step="Scaling",
                    column=column,
                    details=f"Applied {method.capitalize()} to standardize the column.",
                    metrics={},
                    decision=f"Scaling applied: {method}"
                )

        return df_processed
