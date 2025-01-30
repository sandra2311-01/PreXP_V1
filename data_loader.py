import pandas as pd
class DataLoader:
    def __init__(self, file_path, logger=None):
        """
        Initialize the DataLoader.

        Parameters:
            file_path (str): Path to the dataset file.
            logger (ReasoningLogger, optional): Logger instance for recording actions.
        """
        self.file_path = file_path
        self.df = None
        self.logger = logger

    def log_message(self, step, column, details):
        """
        Log a message using the ReasoningLogger.

        Parameters:
            step (str): The preprocessing step.
            column (str): The column being processed (None for general steps).
            details (str): Details about the action taken.
        """
        if self.logger:
            self.logger.log_insight(
                column=column,
                details=details,
                metrics=None
            )

    def load_dataset(self):
        """
        Load the dataset from the specified file path.
        """
        try:
            self.df = pd.read_csv(self.file_path)
            message = f"Dataset loaded successfully with shape {self.df.shape}."
            self.log_message("Dataset Loading", None, message)
        except FileNotFoundError:
            message = f"Error: File not found at {self.file_path}."
            self.log_message("Dataset Loading", None, message)
        except Exception as e:
            message = f"An error occurred while loading the dataset: {e}"
            self.log_message("Dataset Loading", None, message)
