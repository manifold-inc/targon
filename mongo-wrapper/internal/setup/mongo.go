package setup

import (
	"context"
	"fmt"
	"os"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

func InitMongo() (*mongo.Client, error) {
	mongoUsername := os.Getenv("MONGO_USERNAME")
	mongoPassword := os.Getenv("MONGO_PASSWORD")
	mongoHost := os.Getenv("MONGO_HOST")
	mongoPort := os.Getenv("MONGO_PORT")

	var mongoClient *mongo.Client

	var connectionString string
	if mongoUsername != "" && mongoPassword != "" {
		connectionString = fmt.Sprintf("mongodb://%s:%s@%s:%s/%s?authSource=admin&authMechanism=SCRAM-SHA-256",
			mongoUsername, mongoPassword, mongoHost, mongoPort, "targon")
	} else {
		connectionString = fmt.Sprintf("mongodb://%s:%s/%s",
			mongoHost, mongoPort, "targon")
	}

	clientOpts := options.Client().ApplyURI(connectionString)
	client, err := mongo.Connect(clientOpts)
	if err != nil {
		return nil, fmt.Errorf("failed connecting to mongo: %w", err)
	}

	ctx := context.Background()
	err = client.Ping(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("failed pinging mongo: %w", err)
	}

	mongoClient = client
	return mongoClient, nil
}
