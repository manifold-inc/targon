package setup

import (
	"errors"
	"fmt"

	"targon/internal/utils"

	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

func InitMongo() (*mongo.Client, error) {
	MONGO_USERNAME := GetEnv("MONGO_USERNAME", "")
	MONGO_PASSWORD := GetEnv("MONGO_PASSWORD", "")
	MONGO_HOST := GetEnv("MONGO_HOST", "mongo")
	var mongoClient *mongo.Client
	if MONGO_USERNAME == "" || MONGO_PASSWORD == "" {
		return nil, errors.New("no env keys found")
	}
	clientOpts := options.Client().ApplyURI(fmt.Sprintf("mongodb://%s:%s@%s:27017/targon?authSource=admin&authMechanism=SCRAM-SHA-256", MONGO_USERNAME, MONGO_PASSWORD, MONGO_HOST))

	client, err := mongo.Connect(clientOpts)
	if err == nil {
		mongoClient = client
	} else {
		return nil, utils.Wrap("failed connecting to mongo", err)
	}
	return mongoClient, nil
}
