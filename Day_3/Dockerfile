FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    python-dotenv \
    pydantic \
    langchain-core \
    langchain-openai \
    langchain-tavily \
    langgraph \
    numpy

COPY multiagent_lab.py .

RUN mkdir -p outputs

CMD ["python", "multiagent_lab.py"]