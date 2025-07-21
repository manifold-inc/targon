package main

import (
	_ "targon/cli/get"
	"targon/cli/root"
	_ "targon/cli/update"
)

func main() {
	root.Execute()
}
