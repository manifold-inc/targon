package server

import (
	"context"
	"fmt"
	"net/http"
	"mongo-wrapper/internal/setup"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

type AuctionResult struct {
	Timestamp   int64          `bson:"timestamp" json:"timestamp"`
	AuctionData map[string]any `bson:"auction_results,omitempty" json:"auction_data"`
	Block       int            `bson:"block,omitempty" json:"block,omitempty"`
	Weights     *Weights       `bson:"weights,omitempty" json:"weights,omitempty"`
}

type Weights struct {
	UIDs       []uint16  `bson:"uids" json:"uids"`
	Incentives []float64 `bson:"incentives" json:"incentives"`
}

type Server struct {
	echo *echo.Echo
	deps *setup.Dependencies
}

func NewServer(deps *setup.Dependencies) *Server {
	e := echo.New()

	e.Use(middleware.Logger())
	e.Use(middleware.Recover())
	e.Use(middleware.CORS())

	server := &Server{
		echo: e,
		deps: deps,
	}

	server.setupRoutes()

	return server
}

func (s *Server) setupRoutes() {
	s.echo.GET("/api/v1/auction-results", s.getAuctionResults)
}

func (s *Server) Start(addr string) error {
	return s.echo.Start(addr)
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.echo.Shutdown(ctx)
}

func (s *Server) getAuctionResults(c echo.Context) error {
	collection := s.deps.Mongo.Database("targon").Collection("miner_info")

	limit := int64(10)
	if limitStr := c.QueryParam("limit"); limitStr != "" {
		if l, err := fmt.Sscanf(limitStr, "%d", &limit); err != nil || l != 1 {
			limit = 10
		}
	}

	opts := options.Find().SetLimit(limit).SetSort(bson.D{{Key: "timestamp", Value: -1}})

	cursor, err := collection.Find(context.Background(), bson.M{}, opts)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": err.Error(),
		})
	}
	defer cursor.Close(context.Background())

	var auctionResults []AuctionResult
	for cursor.Next(context.Background()) {
		var result AuctionResult
		err := cursor.Decode(&result)
		if err != nil {
			continue
		}
		auctionResults = append(auctionResults, result)
	}

	if err := cursor.Err(); err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": err.Error(),
		})
	}

	return c.JSON(http.StatusOK, auctionResults)
}
