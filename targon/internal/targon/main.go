package targon

import "github.com/subtrahend-labs/gobt/boilerplate"

func SetMainFunc(v *boilerplate.BaseChainSubscriber, c *Core) {
	v.SetMainFunc(func(i <-chan bool, o chan<- bool) {
		mainFunc(c, i, o)
	})
}

func mainFunc(c *Core, i <-chan bool, o chan<- bool) {
	for range i {
		c.Deps.Log.Info("Shuting down...")
		o <- true
		return
	}
}
