FROM golang:alpine AS build

WORKDIR /app

# Copy the Go module files
COPY go.mod .
COPY go.sum .

# Download the Go module dependencies
RUN go mod download

COPY . .

RUN GOOS=linux go build -v ./cmd/targon/targon.go

FROM alpine:3.22.1 
# TODO better way of doing this
ENV VERSION=7.0.0
WORKDIR /app
RUN touch .env
RUN apk add ca-certificates
COPY --from=build /app/targon targon
CMD ["/app/targon"]
