FROM python:3.10

RUN pip install mlflow psycopg2-binary

ENV MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow@database:5432/mlflow

CMD ["mlflow", "server", "--backend-store-uri", "postgresql://mlflow:mlflow@database:5432/mlflow", "--default-artifact-root", "/mlruns", "--host", "0.0.0.0", "--port", "5000"]