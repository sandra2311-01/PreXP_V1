import pandas as pd

class DateHandler:
    def __init__(self, df, insights, logger, date_keywords=None):
        """
        Initialize the DateHandler with the DataFrame, insights, logger, and optional parameters.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            insights (dict): Extracted insights containing data type information.
            logger (ReasoningLogger): Logger instance for recording actions.
            date_keywords (list): Keywords to identify date-related columns.
        """
        self.df = df
        self.insights = insights
        self.logger = logger
        self.date_keywords = date_keywords or ['date', 'time', 'year', 'month', 'day', 'week']
        self.full_date_columns = []
        self.date_component_columns = []

    def detect_and_convert(self):
        """
        Detect and convert date-related columns in the DataFrame.

        Returns:
            pd.DataFrame: Updated DataFrame with converted date columns.
            list: List of full date columns.
            list: List of date component columns.
        """
        month_name_to_number = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }

        for column, details in self.insights['variables'].items():
            dtype = details.get('type', '').lower()

            # Handle datetime-like columns
            if 'datetime' in dtype or dtype.startswith('datetime'):
                try:
                    self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
                    self.full_date_columns.append(column)
                    self.logger.log_preprocess(
                        step="Date Handling",
                        column=column,
                        details="Converted column to datetime format for accurate date processing.",
                        metrics={"data_type": "datetime"},
                        decision="Retained as a full date column"
                    )
                except Exception as e:
                    self.logger.log_preprocess(
                        step="Date Handling",
                        column=column,
                        details=f"Failed to convert column to datetime: {e}",
                        decision="No conversion applied"
                    )
                continue

            # Handle month name columns
            if dtype == 'categorical' and any(keyword in column.lower() for keyword in self.date_keywords):
                unique_values = self.df[column].dropna().unique()
                if any(value in month_name_to_number for value in unique_values):
                    self.df[column] = self.df[column].map(month_name_to_number)
                    self.date_component_columns.append(column)
                    self.logger.log_preprocess(
                        step="Date Handling",
                        column=column,
                        details="Mapped month names to numeric values for consistent processing.",
                        metrics={"unique_values": list(unique_values)},
                        decision="Converted month names to numbers"
                    )
                else:
                    self.logger.log_preprocess(
                        step="Date Handling",
                        column=column,
                        details="Column identified as categorical but did not match month names.",
                        metrics={"unique_values": list(unique_values)},
                        decision="No conversion applied"
                    )
                continue

            # Handle numeric-like date components
            if dtype in ['integer', 'numeric']:
                if any(keyword in column.lower() for keyword in self.date_keywords):
                    self.date_component_columns.append(column)
                    self.logger.log_preprocess(
                        step="Date Handling",
                        column=column,
                        details="Identified column as numeric-like date component.",
                        metrics={"data_type": dtype},
                        decision="Retained as numeric"
                    )
                continue

        # Log final lists
        self.logger.log_preprocess(
            step="Date Handling",
            column=None,
            details="Finalized detection of full date and date component columns.",
            metrics={
                "full_date_columns": self.full_date_columns,
                "date_component_columns": self.date_component_columns
            },
            decision="Completed detection"
        )

        return self.df, self.full_date_columns, self.date_component_columns
