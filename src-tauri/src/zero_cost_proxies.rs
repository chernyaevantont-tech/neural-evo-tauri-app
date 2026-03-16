/// Zero-Cost Proxies for Neural Architecture Search
/// 
/// This module implements SynFlow and other fast metrics for estimating
/// architecture quality without training. Based on ICLR 2021 paper:
/// "Zero-Cost Proxies for Lightweight NAS"
/// 
/// Paper: https://arxiv.org/abs/2101.08134
/// Code: https://github.com/mohsaied/zero-cost-nas

use serde::{Deserialize, Serialize};

/// Configuration for zero-cost proxy evaluation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZeroCostConfig {
    /// Enable zero-cost proxies
    pub enabled: bool,
    /// Strategy: 'two-stage' or 'early-stopping'
    pub strategy: String,
    /// Threshold above which to do full training (0.0-1.0)
    pub fast_pass_threshold: f32,
    /// Epochs for architectures scoring between 0.3-0.7
    pub partial_training_epochs: u32,
    /// Whether to compute multiple metrics or just SynFlow
    pub use_voting: bool,
}

impl Default for ZeroCostConfig {
    fn default() -> Self {
        ZeroCostConfig {
            enabled: false,
            strategy: "two-stage".to_string(),
            fast_pass_threshold: 0.6,
            partial_training_epochs: 20,
            use_voting: false,
        }
    }
}

/// Results from zero-cost proxy evaluation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZeroCostMetrics {
    /// SynFlow score (Σ |∇w ⊙ w|)
    pub synflow: f32,
    /// Normalized score (0.0-1.0)
    pub normalized_score: f32,
    /// Strategy decision
    pub strategy_decision: String, // serialized as snake_case string
}

/// Decision on how to handle the architecture based on zero-cost score
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StrategyDecision {
    /// Skip training entirely, use proxy score as fitness
    Skip,
    /// Do quick partial training
    PartialTrain,
    /// Do full training
    FullTrain,
}

impl ZeroCostMetrics {
    /// Create new metrics from SynFlow score
    pub fn from_synflow(synflow: f32, config: &ZeroCostConfig) -> Self {
        // Normalize score to 0-1 range (empirically observed max SynFlow ~10)
        let normalized = (synflow / 10.0).min(1.0).max(0.0);
        
        let strategy_decision = match config.strategy.as_str() {
            "early-stopping" => {
                if normalized < 0.2 {
                    "skip".to_string()
                } else if normalized < 0.5 {
                    "partial_train".to_string()
                } else {
                    "full_train".to_string()
                }
            }
            _ => {
                // two-stage (default)
                if normalized < config.fast_pass_threshold {
                    if normalized < 0.3 {
                        "skip".to_string()
                    } else {
                        "partial_train".to_string()
                    }
                } else {
                    "full_train".to_string()
                }
            }
        };
        
        ZeroCostMetrics {
            synflow,
            normalized_score: normalized,
            strategy_decision,
        }
    }
    
    /// Get recommended training budget (epochs) based on strategy
    pub fn recommended_epochs(&self, full_epochs: u32, partial_epochs: u32) -> Option<u32> {
        match self.strategy_decision.as_str() {
            "skip" => None,
            "partial_train" => Some(partial_epochs),
            "full_train" => Some(full_epochs),
            _ => Some(full_epochs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_normalization() {
        let config = ZeroCostConfig::default();
        let metrics = ZeroCostMetrics::from_synflow(5.0, &config);
        assert!(metrics.normalized_score > 0.0 && metrics.normalized_score < 1.0);
    }
    
    #[test]
    fn test_two_stage_strategy() {
        let mut config = ZeroCostConfig::default();
        config.strategy = "two-stage".to_string();
        config.fast_pass_threshold = 0.6;
        
        // Low score
        let low = ZeroCostMetrics::from_synflow(2.0, &config);
        assert_eq!(low.strategy_decision, "skip");
        
        // Medium score
        let med = ZeroCostMetrics::from_synflow(4.0, &config);
        assert_eq!(med.strategy_decision, "partial_train");
        
        // High score
        let high = ZeroCostMetrics::from_synflow(8.0, &config);
        assert_eq!(high.strategy_decision, "full_train");
    }
    
    #[test]
    fn test_early_stopping_strategy() {
        let mut config = ZeroCostConfig::default();
        config.strategy = "early-stopping".to_string();
        
        // Very low score
        let very_low = ZeroCostMetrics::from_synflow(1.0, &config);
        assert_eq!(very_low.strategy_decision, "skip");
        
        // Medium score
        let med = ZeroCostMetrics::from_synflow(3.0, &config);
        assert_eq!(med.strategy_decision, "partial_train");
        
        // High score
        let high = ZeroCostMetrics::from_synflow(7.0, &config);
        assert_eq!(high.strategy_decision, "full_train");
    }
    
    #[test]
    fn test_recommended_epochs() {
        let config = ZeroCostConfig::default();
        
        let skip = ZeroCostMetrics {
            synflow: 1.0,
            normalized_score: 0.1,
            strategy_decision: "skip".to_string(),
        };
        assert_eq!(skip.recommended_epochs(50, 20), None);
        
        let partial = ZeroCostMetrics {
            synflow: 4.0,
            normalized_score: 0.4,
            strategy_decision: "partial_train".to_string(),
        };
        assert_eq!(partial.recommended_epochs(50, 20), Some(20));
        
        let full = ZeroCostMetrics {
            synflow: 8.0,
            normalized_score: 0.8,
            strategy_decision: "full_train".to_string(),
        };
        assert_eq!(full.recommended_epochs(50, 20), Some(50));
    }
}
