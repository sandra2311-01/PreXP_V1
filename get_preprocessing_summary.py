def get_preprocessing_summary(logs):
    """
    Generate a concise summary from preprocessing logs.

    Parameters:
        logs (list): Preprocessing logs containing step details.

    Returns:
        dict: A dictionary with steps as keys and summaries as values.
    """
    summary = {}

    for log in logs:
        step = log.get("step", "Unknown Step")
        column = log.get("column", "General")
        decision = log.get("decision", "No decision recorded")

        if step not in summary:
            summary[step] = []

        # Add concise information to the step summary
        summary[step].append(f"Column '{column}': {decision}")

    # Format summaries for display
    formatted_summary = {
        step: "\n".join(actions)
        for step, actions in summary.items()
    }

    return formatted_summary
