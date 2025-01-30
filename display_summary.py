def display_summary(reasoning_logger):
    summary = reasoning_logger.generate_summary()
    for step, logs in summary.items():
        print(f"\n--- {step} ---")
        for log in logs:
            print(f"Column: {log['column']}, Details: {log['details']}, Metrics: {log['metrics']}, Decision: {log['decision']}")
