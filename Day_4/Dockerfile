FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY multiagent_lab_vulnerable.py .
COPY multiagent_lab_hardened.py .

ENV AGENT_FILE=multiagent_lab_hardened.py

CMD ["sh", "-c", "python ${AGENT_FILE}"]