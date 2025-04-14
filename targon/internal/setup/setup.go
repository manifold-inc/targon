package setup

import (
	"os"

	"go.uber.org/zap"
)

type Dependencies struct {
	Log *zap.SugaredLogger
	Env Env
}
type Env struct {
	HOTKEY_PUBLIC_KEY  string
	HOTKEY_PRIVATE_KEY string
	HOTKEY_SS58        string
}

func GetEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

func GetEnvOrPanic(key string, logger *zap.SugaredLogger) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	logger.Panicf("Could not find env key [%s]", key)
	return ""
}

func Init() *Dependencies {
	// Startup
	logger, err := zap.NewProduction()
	if err != nil {
		panic("Failed to get logger")
	}
	sugar := logger.Sugar()

	// Env Variables
	HOTKEY_PUBLIC_KEY := GetEnvOrPanic("HOTKEY_PUBLIC_KEY", sugar)
	HOTKEY_PRIVATE_KEY := GetEnvOrPanic("HOTKEY_PRIVATE_KEY", sugar)
	HOTKEY_SS58 := GetEnvOrPanic("HOTKEY_SS58", sugar)

	return &Dependencies{
		Log: sugar,
		Env: Env{
			HOTKEY_PRIVATE_KEY: HOTKEY_PRIVATE_KEY,
			HOTKEY_PUBLIC_KEY:  HOTKEY_PUBLIC_KEY,
			HOTKEY_SS58:        HOTKEY_SS58,
		},
	}
}
