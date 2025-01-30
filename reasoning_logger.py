class ReasoningLogger:
    def __init__(self):
        self.insight_logs = []  # Logs for extracted insights
        self.preprocess_logs = []  # Logs for preprocessing actions
        self.postprocess_logs = []  # Logs for post-preprocessing actions

    def log_insight(self, column, details, metrics=None):
        """
        Log dataset insights.
        """
        self.insight_logs.append({
            "column": column,
            "details": details,
            "metrics": metrics or {}
        })

    def log_preprocess(self, step, column, details, metrics=None, decision=None):
        """
        Log preprocessing actions.
        """
        self.preprocess_logs.append({
            "step": step,
            "column": column,
            "details": details,
            "metrics": metrics or {},
            "decision": decision
        })



    def get_logs(self, log_type="insight"):
        """
        Retrieve logs based on type.

        Parameters:
            log_type (str): Type of logs to retrieve ('insight', 'preprocess', 'postprocess').

        Returns:
            list: The requested logs.
        """
        if log_type == "insight":
            return self.insight_logs
        elif log_type == "preprocess":
            return self.preprocess_logs
        else:
            raise ValueError("Invalid log type. Choose 'insight', 'preprocess', or 'postprocess'.")

    def generate_summary(self):
        """
        Generate a combined summary of all logs.
        """
        return {
            #"insights": self.insight_logs,
            "preprocessing": self.preprocess_logs,
            #"postprocessing": self.postprocess_logs
        }
