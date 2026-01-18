"""
logging infrastructure for memory agent security research.

this module provides comprehensive logging functionality for the research
framework, including file and console output, log rotation, and structured
logging for experiments, attacks, and defenses.
all comments are lowercase.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


class researchlogger:
    """
    centralized logger for memory agent security research.

    provides structured logging with file rotation, console output,
    and configurable log levels for different components.
    """

    def __init__(self, name: str = "memory_agent_security"):
        """
        initialize the research logger.

        args:
            name: logger name for identification
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # prevent duplicate handlers
        if self.logger.handlers:
            return

        # create logs directory if it doesn't exist
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

        # formatter for structured logging
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # console handler for immediate feedback
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(logging.INFO)
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)

        # file handler with rotation
        self.file_handler = logging.handlers.RotatingFileHandler(
            self.logs_dir / f"{name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def set_level(self, level: str):
        """
        set logging level for all handlers.

        args:
            level: logging level (debug, info, warning, error, critical)
        """
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }

        log_level = level_map.get(level.lower(), logging.INFO)
        self.logger.setLevel(log_level)
        self.console_handler.setLevel(log_level)

    def log_experiment_start(self, experiment_name: str, config: dict):
        """
        log the start of an experiment.

        args:
            experiment_name: name of the experiment
            config: experiment configuration
        """
        self.logger.info(f"starting experiment: {experiment_name}")
        self.logger.debug(f"experiment config: {config}")

    def log_experiment_end(self, exp_name: str, success: bool, duration: float):
        """
        log the end of an experiment.

        args:
            exp_name: name of the experiment
            success: whether experiment completed successfully
            duration: experiment duration in seconds
        """
        status = "completed successfully" if success else "failed"
        self.logger.info(f"experiment {exp_name} {status} in {duration:.2f}s")

    def log_attack_execution(self, attack: str, target: str, success: bool):
        """
        log attack execution results.

        args:
            attack: name of the attack
            target: target memory system
            success: whether attack succeeded
        """
        status = "succeeded" if success else "failed"
        self.logger.info(f"attack {attack} on {target} {status}")

    def log_defense_activation(self, defense: str, attack: str):
        """
        log defense activation against an attack.

        args:
            defense: name of the defense
            attack: name of the attack being defended against
        """
        self.logger.info(f"defense {defense} activated against {attack}")

    def log_error(
        self, component: str, error: Exception, context: Optional[dict] = None
    ):
        """
        log errors with context.

        args:
            component: component where error occurred
            error: exception object
            context: additional context information
        """
        self.logger.error(f"error in {component}: {str(error)}")
        if context:
            self.logger.debug(f"error context: {context}")


# global logger instance
logger = researchlogger()


def get_component_logger(component_name: str) -> researchlogger:
    """
    get a logger instance for a specific component.

    args:
        component_name: name of the component (attacks, defenses,
                        evaluation, etc.)

    returns:
        configured logger for the component
    """
    return researchlogger(f"memory_agent_security.{component_name}")


def setup_experiment_logging(experiment_name: str) -> researchlogger:
    """
    setup logging for a specific experiment.

    creates a separate log file for the experiment and returns
    a configured logger instance.

    args:
        experiment_name: name of the experiment

    returns:
        experiment-specific logger
    """
    exp_logger = researchlogger(f"experiment_{experiment_name}")

    # add experiment-specific file handler
    exp_file_handler = logging.handlers.RotatingFileHandler(
        Path("logs") / f"experiment_{experiment_name}.log",
        maxBytes=50 * 1024 * 1024,  # 50MB for experiments
        backupCount=3,
    )
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(exp_logger.formatter)
    exp_logger.logger.addHandler(exp_file_handler)

    return exp_logger
