FROM flwr_serverapp:0.0.4

WORKDIR /app

RUN pip install avalanche-lib

ENTRYPOINT ["flwr-serverapp"]