package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/aidarkhanov/nanoid"
	_ "github.com/go-sql-driver/mysql"
	"github.com/google/uuid"
	"github.com/labstack/echo/v4"
	"github.com/redis/go-redis/v9"
	"github.com/uptrace/bun/driver/pgdriver"
)

var (
	HUB_SECRET_TOKEN string
	HOTKEY           string
	PUBLIC_KEY       string
	PRIVATE_KEY      string
	INSTANCE_UUID    string
	DEBUG            bool

	client *redis.Client
)

var Reset = "\033[0m"
var Red = "\033[31m"
var Green = "\033[32m"
var Yellow = "\033[33m"
var Blue = "\033[34m"
var Purple = "\033[35m"
var Cyan = "\033[36m"
var Gray = "\033[37m"
var White = "\033[97m"

type Context struct {
	echo.Context
	Info *log.Logger
	Warn *log.Logger
	Err  *log.Logger
}

func main() {
	HOTKEY = safeEnv("HOTKEY")
	PUBLIC_KEY = safeEnv("PUBLIC_KEY")
	PRIVATE_KEY = safeEnv("PRIVATE_KEY")
	HUB_SECRET_TOKEN = safeEnv("HUB_SECRET_TOKEN")
	DB_URL := safeEnv("DB_URL")
	INSTANCE_UUID = uuid.New().String()
	debug, present := os.LookupEnv("DEBUG")

	if !present {
		DEBUG = false
	} else {
		DEBUG, _ = strconv.ParseBool(debug)
	}

	e := echo.New()
	e.Use(func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			reqId, _ := nanoid.Generate("0123456789abcdefghijklmnopqrstuvwxyz", 12)
			InfoLog := log.New(os.Stdout, fmt.Sprintf("%sINFO [%s]: %s", Green, reqId, Reset), log.Ldate|log.Ltime|log.Lshortfile)
			WarnLog := log.New(os.Stdout, fmt.Sprintf("%sWARNING [%s]: %s", Yellow, reqId, Reset), log.Ldate|log.Ltime|log.Lshortfile)
			ErrLog := log.New(os.Stdout, fmt.Sprintf("%sERROR [%s]: %s", Red, reqId, Reset), log.Ldate|log.Ltime|log.Lshortfile)
			cc := &Context{c, InfoLog, WarnLog, ErrLog}
			return next(cc)
		}
	})
	var err error
	db := sql.OpenDB(pgdriver.NewConnector(pgdriver.WithDSN(DB_URL)))
	err = db.Ping()
	if err != nil {
		log.Panicln(err.Error())
	}
	defer db.Close()

	client = redis.NewClient(&redis.Options{
		Addr:     "cache:6379",
		Password: "",
		DB:       0,
	})
	defer client.Close()

	e.POST("/api/chat/completions", func(c echo.Context) error {
		cc := c.(*Context)
		cc.Request().Header.Add("Content-Type", "application/json")
		var req RequestBody
		err = json.NewDecoder(c.Request().Body).Decode(&req)
		if err != nil {
			log.Println("Error decoding json")
			return echo.NewHTTPError(http.StatusInternalServerError, err.Error())
		}
		if req.ApiKey != HUB_SECRET_TOKEN {
			log.Println("Unauthorized request")
			return echo.NewHTTPError(http.StatusUnauthorized, err.Error())
		}
		c.Response().Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		c.Response().Header().Set("Cache-Control", "no-cache")
		c.Response().Header().Set("Connection", "keep-alive")
		c.Response().Header().Set("X-Accel-Buffering", "no")

		cc.Info.Printf("/api/chat/completions\n")
		info, ok := queryMiners(cc, req)
		if ok != nil {
			return c.String(500, ok.Error())
		}
		if(db == nil){
			log.Println("Databse is null outside goroutine")
		}
		go updatOrganicRequest(db, info, req.PubId)
		return c.String(200, "")
	})
	e.Logger.Fatal(e.Start(":80"))
}
