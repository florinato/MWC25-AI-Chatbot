# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the model into the container
COPY ./models/qwen2-0_5b-instruct-q4_0.gguf /app/models/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV NAME World

# Run main_gpt4all.py when the container launches
CMD ["streamlit", "run", "app/main_gpt4all.py"]
