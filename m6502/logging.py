import logging
from rich.logging import RichHandler

class Logger:
    def __init__(self, name="m6502", filename="m6502.log", level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove old handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Rich console handler
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=True,
        )
        console_handler.setLevel(level)

        # File handler
        file_handler = logging.FileHandler(filename, mode="w")
        file_handler.setLevel(level)

        # Formatter (only for file, RichHandler has its own)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger


__all__ = ["Logger"]
