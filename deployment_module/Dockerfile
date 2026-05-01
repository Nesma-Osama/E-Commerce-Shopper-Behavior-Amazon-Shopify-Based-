FROM python:3.11-slim
WORKDIR /app
COPY accelera_deployment/requirements.txt requirements.txt 
RUN python -m pip install --no-cache-dir --prefer-binary --timeout 120 --retries 10 -r requirements.txt
COPY accelera_deployment/server.py server.py
COPY accelera_deployment/modelservice.py modelservice.py
COPY config.json config.json
COPY ./models/knn_model.pkl /app/./models/knn_model.pkl
COPY ./models/linear_regression_weights.pkl /app/./models/linear_regression_weights.pkl
COPY ./models/ordinal_logistic_threshold_weights.pkl /app/./models/ordinal_logistic_threshold_weights.pkl
COPY ./models/preprocessing_pipeline.pkl /app/./models/preprocessing_pipeline.pkl
COPY ./models/target_scaling.pkl /app/./models/target_scaling.pkl
COPY ./models/total_remaining_features.pkl /app/./models/total_remaining_features.pkl
EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
