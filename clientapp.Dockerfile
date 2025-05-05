FROM flwr_clientapp:0.0.3

WORKDIR /app

RUN pip install omegaconf

ENTRYPOINT ["flwr-clientapp"]
