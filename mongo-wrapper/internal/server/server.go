package server

import (
	"context"
	"fmt"
	"net/http"

	"mongo-wrapper/internal/setup"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/subtrahend-labs/gobt/boilerplate"
	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

type MinerBid struct {
	Ip      string  `bson:"ip" json:"ip"`
	Price   int     `bson:"price" json:"price"`
	UID     string  `bson:"uid" json:"uid"`
	Gpus    int     `bson:"gpus" json:"gpus"`
	Payout  float64 `bson:"payout" json:"payout"`
	Diluted bool    `bson:"diluted" json:"diluted"`
}

type Weights struct {
	UIDs       []uint16  `bson:"uids" json:"uids"`
	Incentives []float64 `bson:"incentives" json:"incentives"`
}

type AuctionResult struct {
	Timestamp   int64                  `bson:"timestamp" json:"timestamp"`
	AuctionData map[string][]*MinerBid `bson:"auction_results,omitempty" json:"auction_data"`
	Block       int                    `bson:"block,omitempty" json:"block,omitempty"`
	Weights     *Weights               `bson:"weights,omitempty" json:"weights,omitempty"`
}

type MinerInfoDocument struct {
	Block          int                          `bson:"block,omitempty" json:"block,omitempty"`
	Timestamp      int64                        `bson:"timestamp,omitempty" json:"timestamp,omitempty"`
	AttestErrors   map[string]map[string]string `bson:"attest_errors,omitempty" json:"attest_errors,omitempty"`
	HotkeyToUID    map[string]string            `bson:"hotkey_to_uid,omitempty" json:"hotkey_to_uid,omitempty"`
	AuctionResults map[string][]*MinerBid       `bson:"auction_results,omitempty" json:"auction_results,omitempty"`
	Weights        *Weights                     `bson:"weights,omitempty" json:"weights,omitempty"`
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

	opts := options.Find().SetLimit(limit).SetSort(bson.D{{Key: "timestamp", Value: -1}})

	cursor, err := collection.Find(context.Background(), bson.M{}, opts)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": err.Error(),
		})
	}

	var results []AuctionResult
	err = cursor.All(context.Background(), &results)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": err.Error(),
		})
	}

	return c.JSON(http.StatusOK, results)
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

	opts := options.FindOne().SetSort(bson.D{{Key: "block", Value: -1}})

	var result MinerInfoDocument
	err = collection.FindOne(context.Background(), bson.M{}, opts).Decode(&result)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to get attestation report",
		})
	}

	failed := make(map[string]string)
	if uidErrors, ok := result.AttestErrors[uid]; ok {
		failed = uidErrors
	}

	if result.HotkeyToUID[signedBy] != uid && result.HotkeyToUID[signedBy] != "28" {
		return c.JSON(http.StatusUnauthorized, map[string]string{
			"error": "unauthorized",
		})
	}

	return c.JSON(http.StatusOK, map[string]any{
		"data": failed,
	})
}
