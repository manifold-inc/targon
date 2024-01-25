from loguru import logger


def log_event(self, event):
    # Log event
    if not self.config.neuron.dont_save_events:
        logger.log("EVENTS", "events", **event.__dict__)
