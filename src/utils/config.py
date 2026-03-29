"""
configuration management for memory agent security research.

this module provides utilities for loading, validating, and managing
configuration files for experiments, memory systems, attacks, and defenses.
all comments are lowercase.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


class configmanager:
    """
    centralized configuration management for the research project.

    handles loading of yaml configurations, validation, inheritance,
    and environment variable overrides.
    """

    def __init__(self, config_dir: str = "configs"):
        """
        initialize config manager.

        args:
            config_dir: base directory for configuration files
        """
        self.config_dir = Path(config_dir)
        self._loaded_configs: dict[str, Any] = {}
        self._validators: dict[str, Any] = {}

    def load_config(self, config_path: str, validate: bool = True) -> DictConfig:
        """
        load configuration from yaml file.

        args:
            config_path: relative path to config file from config_dir
            validate: whether to validate the loaded config

        returns:
            loaded configuration as omegaconf dictconfig

        raises:
            filenotfounderror: if config file does not exist
            valueerror: if config validation fails
        """
        full_path = self.config_dir / config_path

        if not full_path.exists():
            raise FileNotFoundError(f"configuration file not found: {full_path}")

        try:
            with open(full_path) as f:
                config_dict = yaml.safe_load(f)

            config = OmegaConf.create(config_dict)

            if validate:
                self._validate_config(config_path, config)

            self._loaded_configs[config_path] = config
            return DictConfig(config)

        except yaml.YAMLError as e:
            raise ValueError(f"invalid yaml in {config_path}: {e}")

    def _validate_config(self, config_path: str, config: DictConfig) -> None:
        """
        validate configuration against registered validators.

        args:
            config_path: path to identify validator
            config: configuration to validate
        """
        validator = self._validators.get(config_path)
        if validator:
            validator(config)

    def register_validator(self, config_path: str, validator: Callable) -> None:
        """
        register a validation function for a config file.

        args:
            config_path: config file path
            validator: function that takes config and raises on invalid
        """
        self._validators[config_path] = validator

    def get_config(self, config_path: str) -> DictConfig | None:
        """
        get previously loaded configuration.

        args:
            config_path: config file path

        returns:
            loaded config or none if not loaded
        """
        return self._loaded_configs.get(config_path)

    def list_configs(self, pattern: str = "*.yaml") -> list[str]:
        """
        list available configuration files.

        args:
            pattern: glob pattern for config files

        returns:
            list of config file paths relative to config_dir
        """
        configs = []
        for yaml_file in self.config_dir.rglob(pattern):
            if yaml_file.is_file():
                configs.append(str(yaml_file.relative_to(self.config_dir)))
        return sorted(configs)


# global config manager instance
config_manager = configmanager()


def load_memory_config(memory_system: str) -> DictConfig:
    """
    load configuration for a specific memory system.

    args:
        memory_system: name of memory system (mem0, amem, memgpt)

    returns:
        memory system configuration
    """
    config_path = f"memory/{memory_system}.yaml"
    return config_manager.load_config(config_path)


def load_experiment_config(experiment_name: str) -> DictConfig:
    """
    load configuration for a specific experiment.

    args:
        experiment_name: name of experiment config file

    returns:
        experiment configuration
    """
    config_path = f"experiments/{experiment_name}.yaml"
    return config_manager.load_config(config_path)
