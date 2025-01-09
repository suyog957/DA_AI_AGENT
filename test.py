# Import the necessary libraries
from transformers import pipeline

# Create a pipeline for text generation
chatbot = pipeline('text-generation', model='gpt2')

# Function to interact with the chatbot
def chat_with_bot():
    print("AI Agent: Hi there! I'm your AI assistant. How can I help you today?")
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if the user wants to exit the chat
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("AI Agent: Goodbye! Have a great day!")
            break
        
        # Generate a response from the chatbot
        response = chatbot(user_input, max_length=100, num_return_sequences=1)
        
        # Print the chatbot's response
        print(f"AI Agent: {response[0]['generated_text']}")

# Main entry point of the program
if __name__ == "__main__":
    chat_with_bot()
