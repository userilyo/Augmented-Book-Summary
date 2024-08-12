# Setup Environment:
- Ensure you have Python 3.8+ installed.
- Install required dependencies using pip install -r requirements.txt (see the full results of "pip freeze" in requirements_dev.txt).

# Run the Application:
- Start the Streamlit app using the command streamlit run app.py.
- The application will load the required models and datasets asynchronously and initialize the retrieval system.

# Using the Application:

- Input: Enter the text you want to summarize in the provided text area. (For test purposes you can use model_test.txt file)
- Generate Summary: Click the "Generate Summary" button to produce both the initial and augmented summaries.
- Evaluate: Enter a reference summary and click the "Evaluate Summaries" button to see the ROUGE scores for both summaries.

# Explore Results:

- The initial and augmented summaries will be displayed, along with their evaluation metrics. 
- Users can compare the quality and informativeness of the summaries and score must better in augmented summary in principles.
