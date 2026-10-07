# Deploying the service

The service ships as a Docker image. Build it with docker build and run it with docker run,
exposing port 8000. Configuration is passed through environment variables. In production we
run two replicas behind a load balancer, and health checks hit the health endpoint every
ten seconds. Rolling deployments replace one replica at a time so that no requests are dropped.