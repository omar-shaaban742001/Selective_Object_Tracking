# Base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Streamlit port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0"]