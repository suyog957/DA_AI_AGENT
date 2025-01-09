# Import the necessary libraries
from phi.agent import Agent  # Used to create AI agents that can interact and perform tasks.
from phi.model.groq import Groq  # This is the model we use to perform AI tasks, like data analysis.
from phi.tools.duckduckgo import DuckDuckGo  # We use DuckDuckGo to search for additional information on the internet if needed.
from dotenv import load_dotenv  # This helps us securely load sensitive information like API keys from a .env file.
import pandas as pd  # Used for data manipulation and analysis
import json  # For handling JSON data
import logging  # For logging and monitoring
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For plotting graphs
import nbformat as nbf  # For creating Jupyter Notebooks

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
            "Ask clarifying questions if needed to understand the data and context",  # The agent should ask for more details if it is unsure.
            "Perform exploratory data analysis (EDA) on the given data",  # The agent should perform EDA.
            "Provide key insights based on the data analysis",  # The agent should provide key insights.
            "Suggest potential prediction model opportunities based on the analysis",  # The agent should suggest prediction model opportunities.
            "Ensure accuracy and responsibility in responses"  # The agent should be careful and reliable.
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
            # "If needed, fetch related data or research papers",  # The team can search for relevant data or research articles.
            # "Suggest potential prediction model opportunities based on the analysis",  # The team should suggest prediction model opportunities.
            # "Ensure information is based on credible sources and is up-to-date",  # Ensure the information is trustworthy and current.
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

# This function performs exploratory data analysis (EDA) on the data and returns a list of code cells for the notebook.
def exploratory_data_analysis(data: pd.DataFrame):
    """
    This function performs EDA on the given data and generates visualizations.
    Returns a list of code cells to be added to the notebook.
    """
    code_cells = []
    
    code_cells.append(f"# Basic Information\nprint({data.info()})")
    code_cells.append(f"# Descriptive Statistics\nprint({data.describe()})")
    
    # Visualization
    sns.pairplot(data)
    plt.savefig('pairplot.png')
    plt.close()
    code_cells.append(f"import matplotlib.pyplot as plt\nplt.figure()\nimg = plt.imread('pairplot.png')\nplt.imshow(img)\nplt.axis('off')\nplt.show()")
    
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.savefig('heatmap.png')
    plt.close()
    code_cells.append(f"plt.figure()\nimg = plt.imread('heatmap.png')\nplt.imshow(img)\nplt.axis('off')\nplt.show()")
    
    # Additional visualizations can be added here
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=data.columns[1], kde=True)  # Change column index as needed
    plt.savefig('histogram.png')
    plt.close()
    code_cells.append(f"plt.figure()\nimg = plt.imread('histogram.png')\nplt.imshow(img)\nplt.axis('off')\nplt.show()")
    
    return code_cells

# This function creates a new Jupyter Notebook with the analysis results.
def create_notebook(data: pd.DataFrame, output_path: str):
    """
    This function creates a new Jupyter Notebook with the results of the data analysis.
    """
    nb = nbf.v4.new_notebook()
    cells = []

    # Title and Introduction
    cells.append(nbf.v4.new_markdown_cell("# Data Analysis Report"))
    cells.append(nbf.v4.new_markdown_cell("## Introduction\nThis notebook provides an analysis of the given dataset, including exploratory data analysis (EDA), key insights, and potential prediction model opportunities."))

    # Basic Information and Descriptive Statistics
    cells.append(nbf.v4.new_code_cell(f"import pandas as pd\n\ndata = pd.read_csv('{output_path}')\nprint(data.info())"))
    cells.append(nbf.v4.new_code_cell(f"print(data.describe())"))

    # Add EDA Visualizations
    eda_cells = exploratory_data_analysis(data)
    for cell in eda_cells:
        cells.append(nbf.v4.new_code_cell(cell))

    # Add Key Insights and Analytics
    cells.append(nbf.v4.new_markdown_cell("## Key Insights and Analytics\nAdd your key insights and analytics here."))

    # Set cells to notebook
    nb['cells'] = cells

    # Write notebook to file
    with open(output_path, 'w') as f:
        nbf.write(nb, f)

# Main entry point of the program. This is where the code starts running.
if __name__ == "__main__":
    # Import the .csv file
    file_path = 'C:/Users/suyog/OneDrive/Desktop/Projects/DA_AI_AGENT/datasets/IMDB/Imdb_Movie_Dataset.csv'  # Replace with your actual file path
    data = import_csv(file_path)
    
    # Create the individual data analysis agent by calling the function `create_data_analysis_agent`.
    data_analysis_agent = create_data_analysis_agent()

    # Create the team of agents by calling the `create_data_analysis_agent_team` function.
    # This includes the data analysis agent we just created.
    agent_team = create_data_analysis_agent_team(data_analysis_agent)

    # Ask the agent team to provide insights based on the given dataset.
    get_data_insights(agent_team, data)
    
    # Create a new Jupyter Notebook with the analysis results
    output_path = 'C:/Users/suyog/OneDrive/Desktop/Projects/DA_AI_AGENT/Result/Data_Analysis_Report.ipynb'
    create_notebook(data, output_path)
