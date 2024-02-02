import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to preprocess text
def preprocess(text):
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords and lemmatize words
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in stopwords.words('english')]
    
    return words

# Create a simple chatbot
def chatbot():
    print("Chatbot: Hi! I'm your chatbot. Type 'exit' to end the conversation.")
    training_data = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = input("Chatbot: What should I respond with? ")
        training_data.append({"input": user_input, "response": response})

    # Save the training data to a JSON file
    with open("training_data.json", "w") as file:
        json.dump(training_data, file)

    print("Chatbot: Thanks for the conversation! Training data saved to 'training_data.json'.")

if __name__ == "__main__":
    chatbot()
