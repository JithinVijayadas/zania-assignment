import os
# from dotenv import load_dotenv

# load_dotenv()

class Config:
    """
    Configuration class to load environment variables.

    This class uses the `dotenv` library to load environment variables from a `.env` file.
    It provides a convenient way to access these variables throughout the application.

    Attributes:
        OPENAI_API_KEY (str): The API key for accessing OpenAI services.
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")