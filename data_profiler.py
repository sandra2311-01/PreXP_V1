from ydata_profiling import ProfileReport
import pickle
from IPython.display import display, HTML
class DataProfiler:
    def __init__(self, df, logger=None, profile_file='profile_report.pkl', report_file='data_report.html'):
        self.df = df
        self.logger = logger
        self.profile_file = profile_file
        self.report_file = report_file
        self.profile = None

    def log_message(self, step, details, column=None, metrics=None, decision=None):
        """
        Log messages based on the step and logger type.
        """
        if self.logger:
            self.logger.log_preprocess(step, column, details, metrics, decision)

    def generate_profile(self):
        """
        Generate and save a profile report for the dataset.
        """
        try:
            self.profile = ProfileReport(self.df, title="Data Analysis Report", explorative=True)
            self.profile.to_file(self.report_file)
            self.log_message(
                step="Profile Generation",
                details=f"Profile report saved to '{self.profile_file}' and '{self.report_file}'."
            )
        except Exception as e:
            self.log_message(
                step="Profile Generation",
                details=f"Failed to generate profile: {e}"
            )

    def load_profile(self):
        """
        Load a previously saved profile report.
        """
        try:
            with open(self.profile_file, "rb") as f:
                self.profile = pickle.load(f)
            self.log_message(
                step="Profile Loading",
                details=f"Profile loaded from '{self.profile_file}'."
            )
        except Exception as e:
            self.log_message(
                step="Profile Loading",
                details=f"Failed to load profile: {e}"
            )

    def display_report(self):
        """
        Display the profile report.
        """
        try:
            with open(self.report_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            display(HTML(html_content))
            self.log_message(
                step="Profile Display",
                details="Profile report displayed successfully."
            )
        except Exception as e:
            self.log_message(
                step="Profile Display",
                details=f"Failed to display profile: {e}"
            )
