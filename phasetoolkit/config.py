"""
Configuration management for phase detection algorithms.

This module provides a centralized way to manage hyperparameters and
configuration options for the phase detection pipeline, making it easy
to experiment with different settings and maintain reproducible results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import os

from .features import DEFAULT_FEATURE_TRANSFORM, FEATURE_TRANSFORMS


@dataclass
class PhaseDetectionConfig:
    """
    Configuration for phase detection algorithms.
    
    This class centralizes all hyperparameters and settings used by the
    phase detection pipeline, making it easy to experiment with different
    configurations and ensure reproducible results.
    """
    
    # Number of phases
    K: int = 2
    
    # HMM parameters
    n_iter: int = 25                    # Maximum EM iterations
    p_stay: float = 0.75                # Probability of staying in current state
    tol: float = 1e-4                   # Convergence tolerance
    random_state: int = 0               # Random seed for reproducibility

    # Phase statistics parameters  
    var_keep: float = 0.95              # Fraction of variance to preserve in PCA
    rmax: int = 16                      # Maximum subspace dimension

    # Feature processing
    feature_transform: str = DEFAULT_FEATURE_TRANSFORM  # Feature transform identifier

    # Data processing parameters
    feature_standardize: bool = False   # Whether to standardize features
    action_standardize: bool = False    # Whether to standardize actions
    
    # Additional metadata
    description: str = ""               # Description of this configuration
    experiment_name: str = ""           # Name for this experiment/run
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'K': self.K,
            'n_iter': self.n_iter,
            'p_stay': self.p_stay,
            'tol': self.tol,
            'random_state': self.random_state,
            'var_keep': self.var_keep,
            'rmax': self.rmax,
            'feature_transform': self.feature_transform,
            'feature_standardize': self.feature_standardize,
            'action_standardize': self.action_standardize,
            'description': self.description,
            'experiment_name': self.experiment_name
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PhaseDetectionConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'PhaseDetectionConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            PhaseDetectionConfig object
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def copy(self, **overrides) -> 'PhaseDetectionConfig':
        """
        Create a copy of this configuration with optional overrides.
        
        Args:
            **overrides: Fields to override in the copy
            
        Returns:
            New PhaseDetectionConfig object
        """
        config_dict = self.to_dict()
        config_dict.update(overrides)
        return self.from_dict(config_dict)
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameters are invalid
        """
        if self.K < 1:
            raise ValueError("K (number of phases) must be positive")
        if self.n_iter < 1:
            raise ValueError("n_iter must be positive")
        if not 0 < self.p_stay < 1:
            raise ValueError("p_stay must be between 0 and 1")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if not 0 < self.var_keep <= 1:
            raise ValueError("var_keep must be between 0 and 1")
        if self.rmax < 1:
            raise ValueError("rmax must be positive")
        if self.feature_transform not in FEATURE_TRANSFORMS:
            available = ', '.join(FEATURE_TRANSFORMS.keys())
            raise ValueError(
                f"Unknown feature transform '{self.feature_transform}'. Available: {available}"
            )


# Predefined configurations for common use cases
DEFAULT_CONFIG = PhaseDetectionConfig()

QUICK_CONFIG = PhaseDetectionConfig(
    K=2,
    n_iter=10,
    p_stay=0.7,
    var_keep=0.9,
    rmax=8,
    description="Quick configuration for fast experimentation"
)

ROBUST_CONFIG = PhaseDetectionConfig(
    K=3,
    n_iter=50,
    p_stay=0.8,
    tol=1e-6,
    var_keep=0.99,
    rmax=32,
    description="Robust configuration for high-quality results"
)

HIGH_PRECISION_CONFIG = PhaseDetectionConfig(
    K=4,
    n_iter=100,
    p_stay=0.75,
    tol=1e-8,
    var_keep=0.995,
    rmax=64,
    feature_standardize=True,
    action_standardize=True,
    description="High-precision configuration for detailed phase detection"
)


def get_config(name: str) -> PhaseDetectionConfig:
    """
    Get a predefined configuration by name.
    
    Args:
        name: Configuration name ('default', 'quick', 'robust', 'high_precision')
        
    Returns:
        PhaseDetectionConfig object
        
    Raises:
        ValueError: If configuration name is not recognized
    """
    configs = {
        'default': DEFAULT_CONFIG,
        'quick': QUICK_CONFIG,
        'robust': ROBUST_CONFIG,
        'high_precision': HIGH_PRECISION_CONFIG
    }
    
    if name not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(f"Unknown configuration '{name}'. Available: {available}")
    
    return configs[name].copy()


def create_config_from_args(**kwargs) -> PhaseDetectionConfig:
    """
    Create configuration from keyword arguments.
    
    This is useful for creating configurations programmatically or from
    command-line arguments.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        PhaseDetectionConfig object
        
    Example:
        config = create_config_from_args(K=3, n_iter=30, p_stay=0.8)
    """
    config = DEFAULT_CONFIG.copy(**kwargs)
    config.validate()
    return config
