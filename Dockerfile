# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 7860

# Set permissions for the /app directory
RUN chmod -R 777 /app

# Command to run the application
# CMD ["chainlit", "run", "app.py", "-w", "--host", "0.0.0.0", "--port", "7860"]
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "7860", "--server.enableXsrfProtection", "false"]

