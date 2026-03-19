"""
logging infrastructure for memory agent security research.

this module provides comprehensive logging functionality for the research
framework, including file and console output, log rotation, and structured
logging for experiments, attacks, defenses, visualization, and benchmarking.

all comments are lowercase.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any


class researchlogger:
    """
    centralized logger for memory agent security research.

    provides structured logging with file rotation, console output,
    and configurable log levels for all framework components including
    attacks, defenses, evaluation, visualization, and experiments.
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
            maxBytes=10 * 1024 * 1024,  # 10mb
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

    # -------------------------------------------------------------------------
    # experiment lifecycle logging
    # -------------------------------------------------------------------------

    def log_experiment_start(self, experiment_name: str, config: Any):
        """
        log the start of an experiment.

        args:
            experiment_name: name of the experiment
            config: experiment configuration (dict, str, or path)
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

    def log_experiment_config_loaded(self, config_file: str, config: dict[str, Any]):
        """
        log that an experiment config was loaded from file.

        args:
            config_file: path to config file
            config: loaded config dictionary
        """
        self.logger.info(f"loaded experiment config from: {config_file}")
        self.logger.debug(f"config contents: {config}")

    def log_experiment_execution_start(
        self, experiment_id: str, num_samples: int, num_trials: int
    ):
        """
        log that an experiment execution is starting.

        args:
            experiment_id: unique identifier
            num_samples: number of content samples
            num_trials: number of evaluation trials
        """
        self.logger.info(
            f"executing experiment {experiment_id}: "
            f"{num_samples} samples x {num_trials} trials"
        )

    def log_experiment_execution_complete(self, experiment_id: str, duration: float):
        """
        log that an experiment execution completed.

        args:
            experiment_id: unique identifier
            duration: execution duration in seconds
        """
        self.logger.info(
            f"experiment {experiment_id} execution complete in {duration:.2f}s"
        )

    def log_experiment_error(self, experiment_id: str, error_msg: str):
        """
        log an experiment execution error.

        args:
            experiment_id: unique identifier
            error_msg: error message string
        """
        self.logger.error(f"experiment {experiment_id} failed: {error_msg}")

    # -------------------------------------------------------------------------
    # batch processing logging
    # -------------------------------------------------------------------------

    def log_batch_progress(self, current: int, total: int, experiment_id: str):
        """
        log batch experiment progress.

        args:
            current: current experiment index (1-based)
            total: total number of experiments
            experiment_id: identifier of current experiment
        """
        self.logger.info(f"batch progress: {current}/{total} - running {experiment_id}")

    def log_batch_complete(self, completed: int, total: int):
        """
        log batch execution completion.

        args:
            completed: number of completed experiments
            total: total number of experiments
        """
        self.logger.info(f"batch complete: {completed}/{total} experiments finished")

    # -------------------------------------------------------------------------
    # results and report logging
    # -------------------------------------------------------------------------

    def log_results_saved(self, output_path: str, num_results: int):
        """
        log that results were saved to disk.

        args:
            output_path: path where results were saved
            num_results: number of result entries saved
        """
        self.logger.info(f"saved {num_results} result(s) to: {output_path}")

    def log_report_generated(self, report_path: str):
        """
        log that a report was generated.

        args:
            report_path: path to the generated report
        """
        self.logger.info(f"report generated: {report_path}")

    # -------------------------------------------------------------------------
    # visualization logging
    # -------------------------------------------------------------------------

    def log_visualization_start(self, visualizer_name: str, output_dir: str):
        """
        log the start of a visualization session.

        args:
            visualizer_name: name of the visualizer class/function
            output_dir: directory where figures will be saved
        """
        self.logger.info(f"visualization started: {visualizer_name} -> {output_dir}")

    def log_visualization_save(self, save_path: str):
        """
        log that a figure was saved.

        args:
            save_path: path where figure was saved
        """
        self.logger.info(f"figure saved: {save_path}")

    def log_visualization_complete(self, plot_name: str, save_path: str):
        """
        log that a visualization was completed and saved.

        args:
            plot_name: name of the plot
            save_path: path where plot was saved
        """
        self.logger.info(f"visualization complete: {plot_name} -> {save_path}")

    def log_visualization_error(self, message: str):
        """
        log a visualization error or warning.

        args:
            message: error or warning message
        """
        self.logger.warning(f"visualization: {message}")

    # -------------------------------------------------------------------------
    # attack and defense logging
    # -------------------------------------------------------------------------

    def log_attack_execution(self, attack: str, target: str, success: bool):
        """
        log attack execution results.

        args:
            attack: name of the attack
            target: target content or memory system
            success: whether attack succeeded
        """
        status = "succeeded" if success else "failed"
        self.logger.info(f"attack {attack} on {target} {status}")

    def log_defense_activation(self, defense: str, attack: Any):
        """
        log defense activation against an attack.

        args:
            defense: name of the defense
            attack: attack info (name string or dict)
        """
        self.logger.info(f"defense {defense} activated against {attack}")

    # -------------------------------------------------------------------------
    # error logging
    # -------------------------------------------------------------------------

    def log_error(
        self,
        component: str,
        error: Exception,
        context: dict | None = None,
    ):
        """
        log errors with context.

        args:
            component: component where error occurred
            error: exception object
            context: additional context information
        """
        self.logger.error(f"error in {component}: {error!s}")
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
        maxBytes=50 * 1024 * 1024,  # 50mb for experiments
        backupCount=3,
    )
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(exp_logger.formatter)
    exp_logger.logger.addHandler(exp_file_handler)

    return exp_logger
