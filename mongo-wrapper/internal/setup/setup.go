package setup

import (
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.uber.org/zap"
)

type Dependencies struct {
	Log   *zap.SugaredLogger
	Mongo *mongo.Client
}

func Init() *Dependencies {
	cfg := zap.NewProductionConfig()
	cfg.Sampling = nil

	logger, err := cfg.Build()
	if err != nil {
		panic("Failed to get logger")
	}
	sugar := logger.Sugar()

	mongoClient, err := InitMongo()
	if err != nil {
		sugar.Fatalw("Failed to initialize MongoDB client", "error", err)
	}

	return &Dependencies{
		Log:   sugar,
		Mongo: mongoClient,
	}
}
