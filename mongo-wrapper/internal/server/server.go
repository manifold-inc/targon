package server

import (
	"context"
	"fmt"
	"net/http"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
	"mongo-wrapper/internal/setup"
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

type AttestationReport struct {
	Failed      map[string]string `bson:"attest_errors" json:"failed"`
	HotkeyToUID map[string]string `bson:"hotkey_to_uid" json:"hotkey_to_uid"`
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
	s.echo.GET("/api/v1/attestation-errors/:uid", s.getAttestationErrors)
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

	opts := options.Find().SetLimit(limit).SetSort(bson.D{{Key: "block", Value: -1}})

	cursor, err := collection.Find(context.Background(), bson.M{}, opts)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": err.Error(),
		})
	}
	defer func() {
		_ = cursor.Close(context.Background())
	}()

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

func (s *Server) getAttestationErrors(c echo.Context) error {
	uid := c.Param("uid")
	if uid == "" {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "No UID provided",
		})
	}

	signedBy := c.Request().Header.Get("Epistula-Signed-By")
	signature := c.Request().Header.Get("Epistula-Request-Signature")
	uuid := c.Request().Header.Get("Epistula-Uuid")
	timestamp := c.Request().Header.Get("Epistula-Timestamp")
	signedFor := c.Request().Header.Get("Epistula-Signed-For")

	if signedBy == "" || signature == "" || uuid == "" || timestamp == "" || signedFor == "" {
		return c.JSON(http.StatusUnauthorized, map[string]string{
			"error": "unauthorized",
		})
	}

	err := boilerplate.VerifyEpistulaHeaders(
		"",
		signature,
		[]byte{},
		timestamp,
		uuid,
		signedFor,
		signedBy,
	)
	if err != nil {
		return c.JSON(http.StatusUnauthorized, map[string]string{
			"error": "unauthorized",
		})
	}

	collection := s.deps.Mongo.Database("targon").Collection("miner_info")

	opts := options.Find().SetLimit(1).SetSort(bson.D{{Key: "block", Value: -1}})

	cursor, err := collection.Find(context.Background(), bson.M{}, opts)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to get attestation report",
		})
	}
	defer func() {
		_ = cursor.Close(context.Background())
	}()

	var results []bson.M
	if err := cursor.All(context.Background(), &results); err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to decode attestation report",
		})
	}

	if len(results) == 0 {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to get attestation report",
		})
	}

	attestErrors, _ := results[0]["attest_errors"].(map[string]any)
	hotkeyToUID, _ := results[0]["hotkey_to_uid"].(map[string]any)

	failed := make(map[string]string)
	if uidErrors, ok := attestErrors[uid].(map[string]any); ok {
		for k, v := range uidErrors {
			if str, ok := v.(string); ok {
				failed[k] = str
			}
		}
	}

	hotkeyToUIDStr := make(map[string]string)
	for k, v := range hotkeyToUID {
		if str, ok := v.(string); ok {
			hotkeyToUIDStr[k] = str
		}
	}

	if hotkeyToUIDStr[signedBy] != uid && hotkeyToUIDStr[signedBy] != "28" {
		return c.JSON(http.StatusUnauthorized, map[string]string{
			"error": "unauthorized",
		})
	}

	return c.JSON(http.StatusOK, map[string]any{
		"data": failed,
	})
}
