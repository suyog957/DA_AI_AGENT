# Import the necessary libraries. These libraries help us build and run the AI-based data analysis agent.
from phi.agent import Agent  # Used to create AI agents that can interact and perform tasks.
from phi.model.groq import Groq  # This is the model we use to perform AI tasks, like data analysis.
from phi.tools.duckduckgo import DuckDuckGo  # We use DuckDuckGo to search for additional information on the internet if needed.
from dotenv import load_dotenv  # This helps us securely load sensitive information like API keys from a .env file.
import pandas as pd  # Used for data manipulation and analysis
import json  # For handling JSON data
import logging  # For logging and monitoring
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For plotting graphs

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Load environment variables (like keys or tokens stored securely) from a file called .env.
load_dotenv()

# This function creates a "Data Analysis Agent" that can analyze data and provide insights.
def create_data_analysis_agent(model: str = "llama-3.3-70b-versatile"):
    """
    This function creates an AI agent designed to help analyze data and provide insights.
    The agent uses a machine learning model and can also search for extra information online.
    """
    return Agent(
        name="Data Analysis Agent",  # This is the name of the agent.
        
        # Here we specify which AI model the agent should use for data analysis.
        model=Groq(id=model),
        
        # The tools the agent can use. Here, DuckDuckGo is the tool we use to search the internet for extra information.
        tools=[DuckDuckGo()],
        
        # Instructions tell the agent how to behave when performing data analysis.
        instructions=[
            "Display basic information about the dataset: number of rows and columns.",  
            "Preview the first few rows of the dataset to understand its structure (columns, data types, and sample values).",  
            "Handle Missing Data: Check for missing values in the dataset.",  
            "If missing data is found, ask whether to: "
                "- Remove rows with missing data, "
                "- Fill missing data with a specific value, "
                "- Drop columns with missing data",
            "Check for duplicate rows in the dataset and handle them appropriately.",
            "Display descriptive statistics of the dataset (mean, median, standard deviation, min, max, etc.).",
            "Visualize the data using appropriate plots (scatter plots, histograms, etc.).",
            "Perform correlation analysis to identify relationships between variables.",
            "Provide key insights based on the data analysis.",
            "Suggest potential prediction model opportunities based on the analysis"
        ],
        
        # Set this to True so we can see when the agent is using its tools (like searching online).
        show_tool_calls=True,
        
        # Set this to True so the agent can provide responses in markdown format, which is a readable style.
        markdown=True
    )

# This function creates a "team" of agents, which includes the data analysis agent and other tools.
def create_data_analysis_agent_team(data_analysis_agent, model: str = "llama-3.3-70b-versatile"):
    """
    This function sets up a team of agents. The team includes the Data Analysis Agent and possibly additional AI models or tools.
    """
    logging.info("Creating data analysis agent team...")
    return Agent(
        # We again specify the AI model the whole team will use. This ensures the team uses the same model for tasks.
        model=Groq(id=model),
        
        # This is where we specify that the team includes the data analysis agent we just created.
        team=[data_analysis_agent],
        
        # Instructions for the team on what they should do when working together.
        instructions=[
            "Perform exploratory data analysis (EDA) on the data",  # The team should perform EDA.
            "Provide key insights based on the data analysis",  # The team should provide key insights.
            "If needed, fetch related data or research papers",  # The team can search for relevant data or research articles.
            "Suggest potential prediction model opportunities based on the analysis",  # The team should suggest prediction model opportunities.
            "Ensure information is based on credible sources and is up-to-date",  # Ensure the information is trustworthy and current.
        ],
        
        # Set this to True to see when the team is using tools (like searching for research papers).
        show_tool_calls=True,
        
        # Set this to True so the team's response is in markdown format (for readability).
        markdown=True
    )

# This function asks the data analysis agent team to analyze the data and provide insights.
def get_data_insights(agent_team, data: pd.DataFrame):
    """
    This function takes a set of data and asks the AI agent team to provide insights based on the data.
    The agent can also search the internet for additional information if needed.
    """
    data_str = data.to_json()
    prompt = f"Based on the following data, provide EDA, key insights, and potential prediction model opportunities: {data_str}"
    # Here we call the agent team to process the data and give a response.
    # The `stream=True` option ensures we get updates in real-time as the agent works.
    agent_team.print_response(prompt, stream=True)

# This function imports a .csv file from a local folder path.
def import_csv(file_path: str) -> pd.DataFrame:
    """
    This function imports a .csv file from the given local folder path and returns a pandas DataFrame.
    """
    return pd.read_csv(file_path)

# This function performs exploratory data analysis (EDA) on the data.
def exploratory_data_analysis(data: pd.DataFrame):
    """
    This function performs EDA on the given data and generates visualizations.
    """
    print("Basic Information")
    print(data.info())
    print("\nDescriptive Statistics")
    print(data.describe())
    
    # Visualization
    sns.pairplot(data)
    plt.show()
    
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.show()

# Main entry point of the program. This is where the code starts running.
if __name__ == "__main__":
    """
    This part of the code is the starting point of the program. It will create the agents and make them work together
    to analyze data and provide insights based on the input provided.
    """
    
    # Import the .csv file
    file_path = 'C:/Users/suyog/OneDrive/Desktop/Projects/DA_AI_AGENT/datasets/IMDB/Imdb_Movie_Dataset.csv'  # Replace with your actual file path
    data = import_csv(file_path)
    
    # Perform exploratory data analysis (EDA)
    exploratory_data_analysis(data)
    
    # Create the individual data analysis agent by calling the function `create_data_analysis_agent`.
    data_analysis_agent = create_data_analysis_agent()

    # Create the team of agents by calling the `create_data_analysis_agent_team` function.
    # This includes the data analysis agent we just created.
    agent_team = create_data_analysis_agent_team(data_analysis_agent)

    # Ask the agent team to provide insights based on the given dataset.
    get_data_insights(agent_team, data)
    
    # You can change the data to test different cases, e.g.:
    # file_path = 'path/to/another/csvfile.csv'  # Another .csv file path
    # data = import_csv(file_path)
    # exploratory_data_analysis(data)
    # get_data_insights(agent_team, data)
