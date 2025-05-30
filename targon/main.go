package main

import (
	"targon/cmd/cli"
	_ "targon/cmd/cli/get"
)

func main() {
	cli.Execute()
}
