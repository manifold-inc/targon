# Mongo Wrapper Service

The Mongo Wrapper Service provides a RESTful HTTP API to access auction results from the MongoDB database used by the Targon validator and miner services.

## Overview

This service acts as a secure middleware layer between external applications and the MongoDB database, providing:

- RESTful API endpoint for auction results
- Health checks and monitoring
- Secure access to auction data only

## Features

### Auction Results Access
- Retrieve auction results with pagination
- Support for query parameters (limit)
- Timestamp and block information included

## API Endpoints

### Auction Results
- `GET /api/v1/auction-results` - Get auction results from the latest intervals

## Configuration

The service is configured via environment variables:

### Environment Variables
- `MONGO_USERNAME` - MongoDB username
- `MONGO_PASSWORD` - MongoDB password
- `MONGO_HOST` - MongoDB host (default: mongo)

## Running the Service

### Using Docker
```bash
# Build the image
docker build -t mongo-wrapper .

# Run the container
docker run -p 8080:8080 \
  -e MONGO_HOST=mongo \
  -e MONGO_USERNAME=your_username \
  -e MONGO_PASSWORD=your_password \
  mongo-wrapper
```

### Using Docker Compose
Add the following service to your `docker-compose.yml`:

```yaml
mongo-wrapper:
  build:
    context: ./mongo-wrapper
    dockerfile: Dockerfile
  ports:
    - "8080:8080"
  environment:
    - MONGO_HOST=mongo
    - MONGO_USERNAME=${MONGO_USERNAME}
    - MONGO_PASSWORD=${MONGO_PASSWORD}
  restart: unless-stopped
```

### Local Development
```bash
# Install dependencies
go mod download

# Run the service
go run cmd/mongo-wrapper/main.go
```

## Query Parameters

### Pagination
- `limit` - Number of auction results to return (default: 10)

### Examples
```
GET /api/v1/auction-results
GET /api/v1/auction-results?limit=5
```

## Response Format

All endpoints return JSON responses with the following structure:

### Success Response
```json
{
  "auction_results": [
    {
      "timestamp": 1234567890,
      "block": 12345,
      "auction_results": {
        "auction_type": [
          {
            "price": 50,
            "uid": "123",
            "gpus": 4,
            "payout": 2.0,
            "diluted": false
          }
        ]
      }
    }
  ],
  "count": 1
}
```

### Error Response
```json
{
  "error": "Error message"
}
```
