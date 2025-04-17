package targon

import (
	"targon/validator"
)

func SetMainFunc(v *validator.BaseValidator, c *Core) {
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
