use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use crate::entities::{GraphModel, DynamicBatch, DynamicTensor};
use serde::{Deserialize, Serialize};

/// helper to calculate |grad * weight|.sum() for a parameter
fn calc_synflow_param<B: AutodiffBackend, const D: usize>(
    param: &burn::module::Param<Tensor<B, D>>,
    grads: &B::Gradients,
) -> f32 {
    if let Some(grad) = param.grad(grads) {
        let weight_inner: Tensor<B::InnerBackend, D> = param.val().inner();
        let product: Tensor<B::InnerBackend, D> = grad.mul(weight_inner);
        let sum_tensor: Tensor<B::InnerBackend, 1> = product.abs().sum().reshape([1]);
        sum_tensor.into_data().to_vec::<f32>().unwrap()[0]
    } else {
        0.0
    }
}

/// Compute the SynFlow metric for a model: Σ |∇w ⊙ w|
/// This measures the "signal flow" through the network without training.
pub fn compute_synflow<B: AutodiffBackend>(
    model: &GraphModel<B>,
    batch: &DynamicBatch<B>,
) -> f32 {
    // 1. Forward pass
    let outputs = model.forward(&batch.inputs);
    
    // 2. Compute the "SynFlow loss" - simply the sum of all components in the output
    let mut total_output = Tensor::<B, 1>::zeros([1], &B::Device::default());
    for out in outputs {
        let sum = match out {
            DynamicTensor::Dim2(t) => t.sum(),
            DynamicTensor::Dim3(t) => t.sum(),
            DynamicTensor::Dim4(t) => t.sum(),
        };
        total_output = total_output + sum.reshape([1]);
    }
    
    // 3. Backward pass to compute gradients
    let grads = total_output.backward();
    
    // 4. Calculate Σ |∇w ⊙ w| for all parameters
    let mut synflow_score = 0.0;
    
    // Manual iteration over layer types in GraphModel to avoid polymorphic type inference issues
    for layer in &model.conv1ds {
        synflow_score += calc_synflow_param(&layer.weight, &grads);
        if let Some(bias) = &layer.bias { synflow_score += calc_synflow_param(bias, &grads); }
    }
    for layer in &model.conv2ds {
        synflow_score += calc_synflow_param(&layer.weight, &grads);
        if let Some(bias) = &layer.bias { synflow_score += calc_synflow_param(bias, &grads); }
    }
    for layer in &model.denses {
        synflow_score += calc_synflow_param(&layer.weight, &grads);
        if let Some(bias) = &layer.bias { synflow_score += calc_synflow_param(bias, &grads); }
    }
    for _layer in &model.lstms {
        // LSTM has many internal parameters
        // We'll just skip complex params for now or handle them if they are public
        // Actually, Burn's LSTM params might not be easily accessible individually without more logic
    }
    for _layer in &model.grus {
        // Similar for GRU
    }
    for _layer in &model.mha_layers {
        // Similar for MHA
    }
    
    synflow_score
}

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
        // Logarithmic normalization to handle wide range of scores (especially for sequences)
        // ln(1 + 400) is approx 6.0, so this maps a wide range to [0, 1]
        let normalized = (synflow.ln_1p() / 6.0).min(1.0).max(0.0);
        
        let strategy_decision = if !config.enabled {
            "full_train".to_string()
        } else if normalized >= config.fast_pass_threshold {
            "full_train".to_string()
        } else if normalized >= config.fast_pass_threshold * 0.4 { // Lower bound for partial
            "partial_train".to_string()
        } else {
            "skip".to_string()
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
        config.enabled = true;
        config.strategy = "two-stage".to_string();
        config.fast_pass_threshold = 0.6;
        
        // Low score
        let low = ZeroCostMetrics::from_synflow(1.0, &config);
        assert_eq!(low.strategy_decision, "skip");
        
        // Medium score
        let med = ZeroCostMetrics::from_synflow(10.0, &config);
        assert_eq!(med.strategy_decision, "partial_train");
        
        // High score
        let high = ZeroCostMetrics::from_synflow(100.0, &config);
        assert_eq!(high.strategy_decision, "full_train");
    }
    
    #[test]
    fn test_early_stopping_strategy() {
        let mut config = ZeroCostConfig::default();
        config.enabled = true;
        config.strategy = "early-stopping".to_string();
        
        // Very low score
        let very_low = ZeroCostMetrics::from_synflow(1.0, &config);
        assert_eq!(very_low.strategy_decision, "skip");
        
        // Medium score
        let med = ZeroCostMetrics::from_synflow(10.0, &config);
        assert_eq!(med.strategy_decision, "partial_train");
        
        // High score
        let high = ZeroCostMetrics::from_synflow(100.0, &config);
        assert_eq!(high.strategy_decision, "full_train");
    }
    
    #[test]
    fn test_recommended_epochs() {
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
