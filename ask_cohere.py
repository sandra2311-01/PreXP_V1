# Initialize Cohere client with your API key
import cohere
import re
cohere_client = cohere.Client('cHWbOI1gxDCJOXofSAFnL41MYiWlGJobYwaLVGZc')


def ask_cohere_about_logs(reasoning_logger, query, verbose=False):
    """
    Queries the LLM (Cohere) about specific preprocessing steps based on logs.

    Parameters:
        reasoning_logger (ReasoningLogger): The logger containing preprocessing logs.
        query (str): The user's query about preprocessing.
        verbose (bool): Whether to include detailed log context in the prompt.

    Returns:
        str: Response from the LLM based on the query and logs.
    """
    # Normalize query and perform context-aware filtering
    query_tokens = re.findall(r'\b\w+\b', query.lower())  # Extract meaningful tokens
    high_priority_steps = ["Encoding", "Scaling", "Missing Data Handling"]  # Define high-priority steps

    relevant_logs = [
        log for log in reasoning_logger.get_logs(log_type="preprocess")
        if any(
            token in (log.get('details', '').lower() or log.get('step', '').lower() or str(log.get('column', '')).lower())
            for token in query_tokens
        ) or log.get('step') in high_priority_steps
    ]

    # If no relevant logs, return a message
    if not relevant_logs:
        return f"No relevant information found for query: {query}"

    # Construct logs context
    logs_context = "\n".join(
        f"Step: {log.get('step')}, Column: {log.get('column')}, Details: {log.get('details')}, Metrics: {log.get('metrics')}, Decision: {log.get('decision')}"
        for log in relevant_logs
    )

    if verbose:
        print("\n--- Logs Context Sent to LLM ---")
        print(logs_context)

    # Simplify the context for clarity
    simplified_context = "\n".join(
        f"{log.get('step')}, Column: {log.get('column')}, Decision: {log.get('decision')}"
        for log in relevant_logs
    )

    # Construct the LLM prompt
    prompt = f"""
    You are an AI assistant helping users analyze preprocessing decisions. Below are logs related to the query:
    {simplified_context}

    User's question: {query}
    Respond in clear, concise bullet points:
    - Highlight each preprocessing step relevant to the query.
    - Explain decisions made, mentioning key metrics where applicable, please be concise and to the point.
    - Provide context only when necessary for clarity.
    - Limit your response to fit in 200 tokens 
    """

    # Call the Cohere API
    response = cohere_client.generate(
        model='command-r-08-2024',
        prompt=prompt,
        max_tokens=200,
        temperature=0.2,  # Reduced temperature for focused responses
    )

    return response.generations[0].text.strip()


#def ask_cohere_about_logs(reasoning_logger, query, verbose=False):
#    """
#    Queries the LLM (Cohere) about specific preprocessing steps based on logs.
#
#    Parameters:
#        reasoning_logger (ReasoningLogger): The logger containing preprocessing logs.
#        query (str): The user's query about preprocessing.
#        verbose (bool): Whether to include detailed log context in the prompt.
#
#    Returns:
#        str: Response from the LLM based on the query and logs.
#    """
#    # Normalize query and perform context-aware filtering
#    query_tokens = re.findall(r'\b\w+\b', query.lower())  # Extract meaningful tokens
#    relevant_logs = [
#        log for log in reasoning_logger.get_logs(log_type="preprocess")
#        if any(
#            token in (str(log.get('details', '')).lower() or str(log.get('step', '')).lower() or str(log.get('column', '')).lower() or str(log.get('decision', '')).lower())
#            for token in query_tokens
#        )
#    ]
#
#    # If no relevant logs, return a message
#    if not relevant_logs:
#        return f"The logs do not provide any information regarding '{query}'. Please refine your query."
#
#    # Construct logs context
#    logs_context = "\n".join(
#        f"Step: {log.get('step')}, Column: {log.get('column')}, Details: {log.get('details')}, Decision: {log.get('decision')}"
#        for log in relevant_logs
#    )
#
#    if verbose:
#        print("\n--- Logs Context Sent to LLM ---")
#        print(logs_context)
#
#    # Simplify the context for clarity
#    simplified_context = "\n".join(
#        f"Step: {log.get('step')}, Column: {log.get('column')}, Decision: {log.get('decision')}"
#        for log in relevant_logs
#    )
#
#    # Construct the LLM prompt
#    prompt = f"""
#    You are an AI assistant helping users analyze preprocessing decisions. Below are logs related to the query:
#    {simplified_context}
#
#    User's question: {query}
#    Provide a concise and accurate response based on the logs. If a column is explicitly mentioned, focus on its preprocessing details
#    """
#
#    # Call the Cohere API
#    response = cohere_client.generate(
#        model='command-r-08-2024',
#        prompt=prompt,
#        max_tokens=200,
#        temperature=0.2,  # Reduced temperature for focused responses
#    )
#
#    # Return the response text
#    return response.generations[0].text.strip()
#