package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"mongo-wrapper/internal/server"
	"mongo-wrapper/internal/setup"
)

func main() {
	deps := setup.Init()
	srv := server.NewServer(deps)

	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		<-sigChan

		if err := srv.Shutdown(context.Background()); err != nil {
			deps.Log.Errorw("Error during server shutdown", "error", err)
		}

		if deps.Mongo != nil {
			if err := deps.Mongo.Disconnect(context.Background()); err != nil {
				deps.Log.Errorw("Error disconnecting from MongoDB", "error", err)
			}
		}
	}()

	if err := srv.Start(fmt.Sprintf(":%s", "8080")); err != nil {
		deps.Log.Fatalw("Failed to start server", "error", err)
	}
}
