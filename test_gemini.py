import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv()


def check_gemini_api_key():
    """
    Checks if the Gemini API key is valid by making a simple request.
    """
    try:
        # Get the API key from the environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Error: GEMINI_API_KEY not found in your .env file.")
            return

        # Configure the generative AI client
        genai.configure(api_key=api_key)

        # Create a model using a valid name from your list
        print("Pinging the Gemini API with a valid model name...")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')  # <-- THIS IS THE UPDATED LINE

        response = model.generate_content("Hello, this is a test.")

        # If we get a response, the key is working
        print("✅ Success! Your API key and model setup are working.")
        print("Gemini's response:", response.text)

    except Exception as e:
        print("❌ Error: The API call failed unexpectedly.")
        print("Detailed error:", e)


if __name__ == "__main__":
    check_gemini_api_key()