FROM flwr_serverapp:0.0.3

WORKDIR /app

RUN pip install omegaconf

ENTRYPOINT ["flwr-serverapp"]