# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements-streamlit.txt .
RUN pip install --no-cache-dir -r requirements-streamlit.txt

# Copy app assets
COPY streamlit_app.py ./
COPY docs/ ./docs/
COPY reports/ ./reports/
COPY artifacts/ ./artifacts/

# Expose port
EXPOSE 8080

# Set Streamlit config
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
