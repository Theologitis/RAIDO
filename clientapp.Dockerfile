FROM flwr_clientapp:0.0.4

WORKDIR /app

RUN pip install avalanche-lib

ENTRYPOINT ["flwr-clientapp"]
