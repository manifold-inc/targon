package setup

import (
	"errors"
	"fmt"

	"github.com/manifold-inc/manifold-sdk/lib/utils"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

func InitMongo() (*mongo.Client, error) {
	MongoUsername := GetEnv("MONGO_USERNAME", "")
	MongoPassword := GetEnv("MONGO_PASSWORD", "")
	MongoHost := GetEnv("MONGO_HOST", "mongo")
	var mongoClient *mongo.Client
	if MongoUsername == "" || MongoPassword == "" {
		return nil, errors.New("no env keys found")
	}
	clientOpts := options.Client().ApplyURI(fmt.Sprintf("mongodb://%s:%s@%s:27017/targon?authSource=admin&authMechanism=SCRAM-SHA-256", MongoUsername, MongoPassword, MongoHost))

	client, err := mongo.Connect(clientOpts)
	if err == nil {
		mongoClient = client
	} else {
		return nil, utils.Wrap("failed connecting to mongo", err)
	}
	return mongoClient, nil
}
