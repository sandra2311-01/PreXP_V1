import sys
import time
import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from reasoning_logger import * 
from data_loader import *
from data_profiler import *
from insights_extractor import *
from missing_data_handler import *
from date_handler import *
from data_encoder import *
from ask_cohere import *
from scaling_and_outlier_handler import *
from get_preprocessing_summary import *
import cohere
import io

# Define navigation steps
steps = [
    "Introduction",
    "Upload Dataset",
    "Data Profiling",
    "Automated Preprocessing",
    "Explainability",
    "Download Data"
]

# Sidebar for navigation
st.sidebar.title("Navigation")
current_step = st.sidebar.radio("Steps", steps)

# Progress bar in sidebar
progress = (steps.index(current_step) + 1) / len(steps)
st.sidebar.progress(progress)

# Step: Introduction
if current_step == "Introduction":
    st.title("Welcome to **PreXP!** \nThe Automated Preprocessing and Explainability Tool!")
    st.markdown("""
    ### Features of the Tool:
    - **Dataset Profiling:** Gain insights into your dataset for informed decision-making.
    - **Automated Preprocessing:** Handle missing data, encoding, scaling, and more.
    - **Explainability:** Query preprocessing decisions with natural language.

    Navigate through the steps using the sidebar to explore the tool.
    """)

# Step: Upload Dataset
elif current_step == "Upload Dataset":
    st.title("Step 1: Upload Your Dataset")
    file = st.file_uploader("Upload your CSV file", type=["csv"])
    if file:
        dataset = pd.read_csv(file)
        st.success("Dataset uploaded successfully!")
        st.write("Here is a preview of your dataset:")
        st.dataframe(dataset.head())
        st.session_state["dataset"] = dataset
        # Ask for target column
        st.markdown("### Specify the Target Column (Optional)")
        target_column = st.text_input("Enter the name of the target column (leave blank if none):")

        # Store target column in session state
        if target_column:
            if target_column in dataset.columns:
                st.session_state["target_column"] = target_column
                st.success(f"Target column set to: {target_column}")
            else:
                st.error("The specified column does not exist in the uploaded dataset.")
        else:
            st.info("No target column specified. General preprocessing will be applied.")

    else:
        st.info("Please upload a dataset to proceed.")



elif current_step == "Data Profiling":
    st.title("Step 2: Data Profiling")

    # Introduction Text
    st.markdown("""
    ### What is Data Profiling?
    Data profiling examines your dataset to extract key insights such as:
    - Statistical summaries
    - Missing data patterns
    - Relationships and correlations between variables

    Relax while we process your data. You'll see live progress updates below.
    """)

    if "dataset" in st.session_state:
        dataset = st.session_state["dataset"]

        if "data_profiler" not in st.session_state:
            st.session_state["data_profiler"] = None

        if st.session_state["data_profiler"] is None:
            try:
                # Redirect sys.stdout to preserve terminal outputs
                original_stdout = sys.stdout
                sys.stdout = open("terminal_logs.txt", "w")

                # Progress Bar in Streamlit
                progress_bar = st.progress(0)
                progress_status = st.empty()

                # Profiling steps (Streamlit & Terminal)
                with st.spinner("Profiling in progress..."):
                    # Step 1: Summarize Dataset
                    progress_status.text("Summarizing dataset...")
                    print("Terminal Output: Summarizing dataset - Starting...")
                    profile = ProfileReport(dataset, title="Data Profiling Report")
                    progress_bar.progress(25)
                    print("Terminal Output: Summarizing dataset - 100% Completed!")

                    # Step 2: Generate Report Structure
                    progress_status.text("Generating report structure...")
                    print("Terminal Output: Generating report structure - Starting...")
                    profile.to_file("data_report.html")
                    progress_bar.progress(50)
                    print("Terminal Output: Generating report structure - 100% Completed!")

                    # Step 3: Render HTML
                    progress_status.text("Rendering HTML report...")
                    print("Terminal Output: Rendering HTML report - Starting...")
                    progress_bar.progress(75)
                    print("Terminal Output: Rendering HTML report - 100% Completed!")

                    # Step 4: Export Report
                    progress_status.text("Exporting report to file...")
                    print("Terminal Output: Exporting report to file - Starting...")
                    progress_bar.progress(100)
                    print("Terminal Output: Exporting report to file - 100% Completed!")

                st.success("Data profiling completed successfully! 🎉")

                # Restore sys.stdout
                sys.stdout.close()
                sys.stdout = original_stdout

                # Store profiler in session state
                st.session_state["data_profiler"] = profile

                # Optional: Button to view the profiling report
                st.markdown("### View Profiling Report")
                if st.button("Show Report"):
                    st_profile_report(profile)

            except Exception as e:
                # Restore stdout in case of error
                sys.stdout = original_stdout
                st.error(f"Error during profiling: {e}")
                print(f"Terminal Output: Error during profiling: {e}")
        else:
            st.success("Data profiling has already been completed.")
            st.markdown("### View Existing Profiling Report")
            if st.button("Show Report"):
                st_profile_report(st.session_state["data_profiler"])

    else:
        st.error("Please upload a dataset first.")





elif current_step == "Automated Preprocessing":
    st.title("Step 3: Automated Preprocessing")

    if "data_profiler" in st.session_state:
        data_profiler = st.session_state["data_profiler"]
        dataset = st.session_state["dataset"]
        target_column = st.session_state.get("target_column")  # Fetch target column if it exists

        reasoning_logger = ReasoningLogger()

        # Extract Insights
        st.info("Extracting insights...")
        insights_extractor = InsightsExtractor(profile=data_profiler, logger=reasoning_logger)
        all_insights = insights_extractor.extract_description()

        # Log extracted insights
        for key, insights in all_insights.items():
            try:
                if isinstance(insights, dict):
                    for column, info in insights.items():
                        reasoning_logger.log_insight(
                            column=column,
                            details=f"Extracted insights from '{key}' for column '{column}'.",
                            metrics=info
                        )
                elif isinstance(insights, list):
                    for i, item in enumerate(insights):
                        reasoning_logger.log_insight(
                            column=f"{key}[{i}]",
                            details=f"Extracted insight from '{key}' (item {i}).",
                            metrics={"value": item}
                        )
                else:
                    reasoning_logger.log_insight(
                        column=key,
                        details=f"General insight extracted for '{key}'.",
                        metrics={"value": insights}
                    )
            except Exception as e:
                reasoning_logger.log_insight(
                    column=key,
                    details=f"Error processing insights for '{key}': {e}"
                )

        # Preprocessing Steps
        st.info("Handling missing data...")
        missing_data_handler = MissingDataHandler(
            insight_logs=reasoning_logger.get_logs(log_type="insight"),
            logger=reasoning_logger
        )
        missing_data_categories = missing_data_handler.categorize_missing_data(dataset)
        df_cleaned = missing_data_handler.handle_missing_data(dataset, missing_data_categories)

        st.info("Handling date-related columns...")
        date_handler = DateHandler(df=df_cleaned, insights=all_insights, logger=reasoning_logger)
        df_with_dates, _, _ = date_handler.detect_and_convert()

        st.info("Encoding categorical variables...")
        encoder = DataEncoder(df=df_with_dates, insights=all_insights, logger=reasoning_logger)
        encoded_df = encoder.encode_columns(target_column=target_column)

        # Add Scaling and Outlier Handling
        st.info("Applying scaling and outlier handling...")
        handler = ScalingAndOutlierHandler(df=encoded_df, insights=all_insights, logger=reasoning_logger)
        scaling_decisions, outlier_decisions = handler.determine_scaling_and_outlier_handling()
        df_processed = handler.apply_scaling_and_outliers(scaling_decisions, outlier_decisions)

        # Save processed DataFrame and logger to session state
        st.session_state["encoded_df"] = df_processed
        st.session_state["reasoning_logger"] = reasoning_logger


        st.success("Preprocessing completed successfully!")
        st.write("Here is your preprocessed dataset:")
        st.dataframe(df_processed.head())
    else:
        st.error("Please complete the data profiling step first.")


# Step: Explainability
#elif current_step == "Explainability":
#    st.title("Step 4: Explainability")
#
#    # Check if the preprocessing logs and dataset exist
#    if "encoded_df" in st.session_state and "reasoning_logger" in st.session_state:
#        reasoning_logger = st.session_state["reasoning_logger"]
#
#        st.markdown("""
#        ### Query Preprocessing Decisions
#        Ask questions about preprocessing steps and understand what was done to your data.
#        """)
#
#        query = st.text_input("Ask about a preprocessing step (e.g., 'What was done to column X?')")
#        if query:
#            try:
#                response = ask_cohere_about_logs(reasoning_logger, query)
#                st.write("### Response:")
#                st.success(response)
#            except Exception as e:
#                st.error(f"An error occurred while querying the logs: {e}")
#
#        # Button to display preprocessing summary
#        st.markdown("### Preprocessing Summary")
#        if st.button("Show Preprocessing Summary"):
#            preprocess_logs = reasoning_logger.get_logs(log_type="preprocess")
#            if preprocess_logs:
#                summary = get_preprocessing_summary(preprocess_logs)
#                for step, decision in summary.items():
#                    st.subheader(step)
#                    st.text(decision)
#            else:
#                st.warning("No preprocessing summary is available.")
#    else:
#        st.error("Please complete the preprocessing step first.")
# Step: Explainability
# Step: Explainability
# Step: Explainability
# Step: Explainability
# Step: Explainability
# Step: Explainability
# Step: Explainability
# Step: Explainability
elif current_step == "Explainability":
    st.title("Step 4: Explainability")

    # Check if logs and dataset exist
    if "encoded_df" in st.session_state and "reasoning_logger" in st.session_state and "dataset" in st.session_state:
        reasoning_logger = st.session_state["reasoning_logger"]
        dataset_before_encoding = st.session_state["dataset"]  # Original dataset before encoding
        dataset_after_encoding = st.session_state["encoded_df"]  # Processed dataset after encoding

        st.markdown("### Query Preprocessing Decisions")
        query = st.text_input("Ask about a preprocessing step (e.g., 'What was done to column X?')")
        if query:
            try:
                response = ask_cohere_about_logs(reasoning_logger, query)
                st.write("### Response:")
                st.success(response)
            except Exception as e:
                st.error(f"An error occurred while querying the logs: {e}")

        # Get Preprocessing Logs
        preprocess_logs = reasoning_logger.get_logs(log_type="preprocess")

        if preprocess_logs:
            import plotly.express as px

            # Convert logs to DataFrame
            logs_df = pd.DataFrame(preprocess_logs)

            ## --- 1️⃣ Missing Data Handling ---
            st.markdown("### **Missing Data Handling**")
            missing_logs = logs_df[logs_df["step"] == "Missing Data Categorization"]

            if not missing_logs.empty:
                missing_summary = missing_logs.groupby("decision")["column"].apply(list).reset_index()
                missing_summary["Count"] = missing_summary["column"].apply(len)

                # Replace abbreviations with full descriptions
                missing_summary["decision"] = missing_summary["decision"].replace({
                    "MCAR": "Missing Completely at Random (MCAR)",
                    "MAR": "Missing at Random (MAR)",
                    "MNAR": "Missing Not at Random (MNAR)"
                })

                fig_missing = px.pie(
                    missing_summary,
                    values="Count",
                    names="decision",
                    title="How Missing Data Was Handled",
                    hover_data={"column": True}
                )
                fig_missing.update_traces(textinfo="percent+label", hoverinfo="label+percent+text")
                st.plotly_chart(fig_missing)

            ## --- 2️⃣ Date Handling as a Bar Chart ---
            st.markdown("### **Date Handling**")
            date_logs = logs_df[logs_df["step"] == "Date Handling"]

            if not date_logs.empty:
                date_summary = date_logs.groupby("decision")["column"].apply(list).reset_index()
                date_summary["Count"] = date_summary["column"].apply(len)

                fig_date = px.bar(
                    date_summary,
                    x="decision",
                    y="Count",
                    title="Types of Date Transformations Applied",
                    labels={"decision": "Date Transformation", "Count": "Number of Columns"},
                    text_auto=True
                )
                st.plotly_chart(fig_date)

            ## --- 3️⃣ Encoding (Pie Chart & Interactive Selection) ---
            st.markdown("### **Encoding Methods Overview**")
            encoding_logs = logs_df[logs_df["step"] == "Encoding"]

            if not encoding_logs.empty:
                encoding_summary = encoding_logs.groupby("decision")["column"].apply(list).reset_index()
                encoding_summary["Count"] = encoding_summary["column"].apply(len)

                fig_encoding = px.pie(
                    encoding_summary,
                    values="Count",
                    names="decision",
                    title="Encoding Methods Applied",
                    hover_data={"column": True}
                )
                fig_encoding.update_traces(textinfo="percent+label", hoverinfo="label+percent+text")
                st.plotly_chart(fig_encoding)

            ## --- Interactive Encoding Selection ---
            st.markdown("### **Encoding Details per Column**")
            categorical_columns = dataset_before_encoding.select_dtypes(include=["object"]).columns.tolist()
            selected_column = st.selectbox("Select a column to see its encoding technique:", categorical_columns)

            if selected_column:
                encoding_details = encoding_logs[encoding_logs["column"] == selected_column]
                if not encoding_details.empty:
                    encoding_type = encoding_details["decision"].values[0]

                    st.write(f"#### **Encoding Applied to `{selected_column}`:** {encoding_type}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"#### **Before Encoding** ({selected_column})")
                        st.dataframe(dataset_before_encoding[[selected_column]].head())

                    transformed_cols = [col for col in dataset_after_encoding.columns if col.startswith(selected_column)]
                    with col2:
                        st.write(f"#### **After Encoding** ({selected_column})")
                        if transformed_cols:
                            st.dataframe(dataset_after_encoding[transformed_cols].head())
                        else:
                            st.write("Column unchanged after processing.")
                else:
                    st.warning("No encoding applied to this column.")

            ## --- 4️⃣ Scaling & Outlier Handling (Pie Chart & Interactive Selection) ---
            st.markdown("### **Scaling & Outlier Handling Overview**")
            scaling_logs = logs_df[logs_df["step"] == "Scaling and Outlier Handling"]

            if not scaling_logs.empty:
                scaling_summary = scaling_logs.groupby("decision")["column"].apply(list).reset_index()
                scaling_summary["Count"] = scaling_summary["column"].apply(len)

                fig_scaling = px.pie(
                    scaling_summary,
                    values="Count",
                    names="decision",
                    title="Scaling & Outlier Handling Methods Applied",
                    hover_data={"column": True}
                )
                fig_scaling.update_traces(textinfo="percent+label", hoverinfo="label+percent+text")
                st.plotly_chart(fig_scaling)

            ## --- Interactive Scaling Selection ---
            st.markdown("### **Scaling & Outlier Handling per Column**")
            numerical_columns = dataset_before_encoding.select_dtypes(include=["int64", "float64"]).columns.tolist()
            selected_scaling_column = st.selectbox("Select a column to see its scaling technique:", numerical_columns)

            if selected_scaling_column:
                scaling_details = scaling_logs[scaling_logs["column"] == selected_scaling_column]
                if not scaling_details.empty:
                    scaling_type = scaling_details["decision"].values[0]

                    st.write(f"#### **Scaling Applied to `{selected_scaling_column}`:** {scaling_type}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"#### **Before Scaling** ({selected_scaling_column})")
                        st.dataframe(dataset_before_encoding[[selected_scaling_column]].head())

                    with col2:
                        st.write(f"#### **After Scaling** ({selected_scaling_column})")
                        st.dataframe(dataset_after_encoding[[selected_scaling_column]].head())

                else:
                    st.warning("No scaling applied to this column.")

            ## --- 5️⃣ Display **Detailed Logs in a Table** ---
            st.markdown("### **Detailed Preprocessing Logs**")
            st.dataframe(logs_df[["step", "column", "decision", "details"]])

        else:
            st.warning("No preprocessing summary is available.")
    else:
        st.error("Please complete the preprocessing step first.")


# Step: Download Data
elif current_step == "Download Data":
    st.title("Step 5: Download Preprocessed Data")
    if "encoded_df" in st.session_state:
        encoded_df = st.session_state["encoded_df"]
        csv = encoded_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Preprocessed Dataset as CSV",
            data=csv,
            file_name="preprocessed_dataset.csv",
            mime="text/csv"
        )
    else:
        st.error("Please complete the preprocessing step first.")
