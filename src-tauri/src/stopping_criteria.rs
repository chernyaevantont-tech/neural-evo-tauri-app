/// Stopping Criteria Module
/// 
/// Implements flexible stopping criteria for evolution with any/all policy:
/// - GenerationLimit: Stop after N generations
/// - FitnessPlateau: Stop if fitness improvement < threshold for patience generations
/// - TimeLimit: Stop after elapsed time exceeds limit
/// - TargetAccuracy: Stop if best accuracy reaches threshold
/// - ManualStop: Triggered via external signal
///
/// Usage:
/// ```rust
/// let state = EvolutionProgressState::new();
/// state.on_generation_complete(generation, best_fitness, best_accuracy);
/// let decision = check_stopping_criteria(&criteria, &state, policy);
/// if decision.should_stop {
///     println!("Stopped: {}", decision.reason_message);
/// }
/// ```

use crate::dtos::StoppingCriterion;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// Tracks evolution progress state for criterion evaluation
#[derive(Debug, Clone)]
pub struct EvolutionProgressState {
    /// Current generation number (0-indexed)
    pub generation: u32,
    
    /// Best fitness found so far
    pub best_fitness: f32,
    
    /// Best accuracy found so far
    pub best_accuracy: f32,
    
    /// Population average fitness
    pub population_avg_fitness: f32,
    
    /// Pareto front coverage (0.0-1.0)
    pub pareto_coverage: f32,
    
    /// Time elapsed since evolution start
    pub elapsed_time: Duration,
    
    /// Evolution start time (for time limit calculation)
    start_time: Instant,
    
    /// History of best fitness per generation (for plateau detection)
    fitness_history: VecDeque<f32>,
    
    /// History of best accuracy per generation
    accuracy_history: VecDeque<f32>,
    
    /// Highest fitness improvement threshold for plateau detection
    /// Stores the best delta between consecutive generations
    max_improvement_seen: f32,
}

impl EvolutionProgressState {
    /// Create new evolution progress state
    pub fn new() -> Self {
        EvolutionProgressState {
            generation: 0,
            best_fitness: f32::NEG_INFINITY,
            best_accuracy: 0.0,
            population_avg_fitness: 0.0,
            pareto_coverage: 0.0,
            elapsed_time: Duration::ZERO,
            start_time: Instant::now(),
            fitness_history: VecDeque::with_capacity(1000),
            accuracy_history: VecDeque::with_capacity(1000),
            max_improvement_seen: 0.0,
        }
    }
    
    /// Update state after a generation completes
    pub fn on_generation_complete(
        &mut self,
        generation: u32,
        best_fitness: f32,
        best_accuracy: f32,
        population_avg_fitness: f32,
        pareto_coverage: f32,
    ) {
        self.generation = generation;
        
        // Track improvement
        let fitness_improvement = (best_fitness - self.best_fitness).abs();
        if fitness_improvement > self.max_improvement_seen {
            self.max_improvement_seen = fitness_improvement;
        }
        
        self.best_fitness = best_fitness.max(self.best_fitness);
        self.best_accuracy = best_accuracy.max(self.best_accuracy);
        self.population_avg_fitness = population_avg_fitness;
        self.pareto_coverage = pareto_coverage.clamp(0.0, 1.0);
        
        self.elapsed_time = self.start_time.elapsed();
        
        // Maintain history for plateau detection
        self.fitness_history.push_back(best_fitness);
        self.accuracy_history.push_back(best_accuracy);
        
        // Keep history bounded (max 1000 generations)
        if self.fitness_history.len() > 1000 {
            self.fitness_history.pop_front();
            self.accuracy_history.pop_front();
        }
    }
    
    /// Reset state (for new runs)
    pub fn reset(&mut self) {
        *self = EvolutionProgressState::new();
    }
    
    /// Get improvement between last N generations
    fn get_improvement_last_n(&self, n: u32) -> f32 {
        let n = (n as usize).min(self.fitness_history.len());
        if n < 2 {
            return 0.0;
        }
        
        let oldest_fitness = self.fitness_history[self.fitness_history.len() - n];
        let newest_fitness = self.fitness_history[self.fitness_history.len() - 1];
        
        (newest_fitness - oldest_fitness).abs()
    }
    
    /// Check if best fitness has plateaued over last N generations
    fn is_fitness_plateaued(&self, patience: u32, threshold: f32) -> bool {
        if self.fitness_history.len() < (patience as usize + 1) {
            return false;
        }
        
        let improvement = self.get_improvement_last_n(patience);
        improvement < threshold
    }
    
    /// Check if best accuracy has plateaued over last N generations
    fn is_accuracy_plateaued(&self, patience: u32, threshold: f32) -> bool {
        if self.accuracy_history.len() < (patience as usize + 1) {
            return false;
        }
        
        let n = (patience as usize).min(self.accuracy_history.len());
        let oldest_accuracy = self.accuracy_history[self.accuracy_history.len() - n];
        let newest_accuracy = self.accuracy_history[self.accuracy_history.len() - 1];
        
        (newest_accuracy - oldest_accuracy).abs() < threshold
    }
    
    /// Get population average fitness for plateau detection
    /// (May be used for future monitoring of population-level stopping criteria)
    #[allow(dead_code)]
    fn get_avg_fitness_last_n(&self, n: u32) -> f32 {
        let n = (n as usize).min(self.fitness_history.len());
        if n == 0 {
            return 0.0;
        }
        
        let sum: f32 = self.fitness_history.iter()
            .rev()
            .take(n)
            .sum();
        sum / n as f32
    }
}

impl Default for EvolutionProgressState {
    fn default() -> Self {
        Self::new()
    }
}

/// Decision result from stopping criteria evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoppingDecision {
    /// Whether evolution should stop
    pub should_stop: bool,
    
    /// Which criteria triggered stopping (if should_stop=true)
    pub triggered_criteria: Vec<String>,
    
    /// Human-readable reason for stopping
    pub reason_message: String,
}

impl Default for StoppingDecision {
    fn default() -> Self {
        StoppingDecision {
            should_stop: false,
            triggered_criteria: Vec::new(),
            reason_message: String::new(),
        }
    }
}

/// Check stopping criteria and return decision
///
/// # Arguments
/// * `criteria` - List of stopping criteria to check
/// * `state` - Current evolution progress state
/// * `policy` - Policy type: "any" (stop if ANY criterion triggered) or "all" (stop if ALL criteria triggered)
/// * `manual_stop_requested` - Whether manual stop was requested
///
/// # Returns
/// StoppingDecision with should_stop flag and reason message
pub fn check_stopping_criteria(
    criteria: &[StoppingCriterion],
    state: &EvolutionProgressState,
    policy: &str,
    manual_stop_requested: bool,
) -> StoppingDecision {
    let mut triggered = Vec::new();
    let mut reasons = Vec::new();
    
    for criterion in criteria {
        match criterion {
            StoppingCriterion::GenerationLimit { max_generations } => {
                if state.generation >= *max_generations {
                    triggered.push("GenerationLimit".to_string());
                    reasons.push(format!(
                        "Generation limit reached ({} >= {})",
                        state.generation, max_generations
                    ));
                }
            }
            
            StoppingCriterion::FitnessPlateau {
                patience_generations,
                improvement_threshold,
                monitor,
            } => {
                let is_plateaued = match monitor.as_str() {
                    "best_fitness" => {
                        state.is_fitness_plateaued(*patience_generations, *improvement_threshold)
                    }
                    "best_accuracy" => {
                        state.is_accuracy_plateaued(*patience_generations, *improvement_threshold)
                    }
                    "population_avg" => {
                        // For population avg, check if last N generations had < threshold improvement
                        let improvement = state.get_improvement_last_n(*patience_generations);
                        improvement < *improvement_threshold
                    }
                    _ => false,
                };
                
                if is_plateaued {
                    triggered.push("FitnessPlateau".to_string());
                    reasons.push(format!(
                        "Fitness plateau detected ({} metric no improvement > {} for {} generations)",
                        monitor, improvement_threshold, patience_generations
                    ));
                }
            }
            
            StoppingCriterion::TimeLimit { max_seconds } => {
                let elapsed_secs = state.elapsed_time.as_secs_f32();
                if elapsed_secs >= *max_seconds as f32 {
                    triggered.push("TimeLimit".to_string());
                    reasons.push(format!(
                        "Time limit exceeded ({:.1}s >= {}s)",
                        elapsed_secs, max_seconds
                    ));
                }
            }
            
            StoppingCriterion::TargetAccuracy { threshold } => {
                if state.best_accuracy >= *threshold {
                    triggered.push("TargetAccuracy".to_string());
                    reasons.push(format!(
                        "Target accuracy reached ({:.2}% >= {:.2}%)",
                        state.best_accuracy * 100.0,
                        threshold * 100.0
                    ));
                }
            }
            
            StoppingCriterion::ManualStop => {
                if manual_stop_requested {
                    triggered.push("ManualStop".to_string());
                    reasons.push("Manual stop requested by user".to_string());
                }
            }
        }
    }
    
    // Apply policy
    let should_stop = match policy.to_lowercase().as_str() {
        "any" => !triggered.is_empty(),
        "all" => triggered.len() == criteria.len() && !triggered.is_empty(),
        _ => false, // Unknown policy defaults to no stop
    };
    
    StoppingDecision {
        should_stop,
        triggered_criteria: triggered.clone(),
        reason_message: if should_stop {
            format!("Evolution stopped: {}", reasons.join("; "))
        } else {
            String::new()
        },
    }
}

/// Validate stopping criteria configuration for correctness
pub fn validate_stopping_config(
    criteria: &[StoppingCriterion],
    policy: &str,
) -> Result<(), String> {
    if criteria.is_empty() {
        return Err("At least one stopping criterion must be specified".to_string());
    }
    
    let valid_policies = ["any", "all"];
    if !valid_policies.contains(&policy.to_lowercase().as_str()) {
        return Err(format!(
            "Invalid policy: '{}'. Must be 'any' or 'all'",
            policy
        ));
    }
    
    // Validate individual criteria
    for criterion in criteria {
        match criterion {
            StoppingCriterion::GenerationLimit { max_generations } => {
                if *max_generations == 0 {
                    return Err(
                        "GenerationLimit must be > 0".to_string()
                    );
                }
            }
            
            StoppingCriterion::FitnessPlateau {
                patience_generations,
                improvement_threshold,
                monitor,
            } => {
                if *patience_generations == 0 {
                    return Err("FitnessPlateau patience must be > 0".to_string());
                }
                if *improvement_threshold < 0.0 {
                    return Err("FitnessPlateau threshold must be >= 0".to_string());
                }
                let valid_monitors = ["best_fitness", "best_accuracy", "population_avg"];
                if !valid_monitors.contains(&monitor.as_str()) {
                    return Err(format!(
                        "Invalid monitor metric: '{}'. Must be one of: {:?}",
                        monitor, valid_monitors
                    ));
                }
            }
            
            StoppingCriterion::TimeLimit { max_seconds } => {
                if *max_seconds == 0 {
                    return Err("TimeLimit must be > 0 seconds".to_string());
                }
            }
            
            StoppingCriterion::TargetAccuracy { threshold } => {
                if *threshold < 0.0 || *threshold > 1.0 {
                    return Err("TargetAccuracy threshold must be between 0.0 and 1.0".to_string());
                }
            }
            
            StoppingCriterion::ManualStop => {
                // No validation needed
            }
        }
    }
    
    Ok(())
}

/// Generate preview of stopping criteria evaluation
/// Useful for showing estimated stopping time to user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoppingPreview {
    /// Estimated generations to stopping
    pub estimated_generations: Vec<u32>,
    
    /// Stopping criteria description
    pub criterion_descriptions: Vec<String>,
    
    /// Policy explanation
    pub policy_explanation: String,
}

/// Generate a preview of how stopping criteria will work
pub fn generate_stopping_preview(
    criteria: &[StoppingCriterion],
    policy: &str,
) -> Result<StoppingPreview, String> {
    validate_stopping_config(criteria, policy)?;
    
    let mut estimated_generations = Vec::new();
    let mut descriptions = Vec::new();
    
    for criterion in criteria {
        match criterion {
            StoppingCriterion::GenerationLimit { max_generations } => {
                estimated_generations.push(*max_generations);
                descriptions.push(format!(
                    "Stop after {} generations",
                    max_generations
                ));
            }
            
            StoppingCriterion::FitnessPlateau {
                patience_generations,
                improvement_threshold,
                monitor,
            } => {
                estimated_generations.push(patience_generations * 2);
                descriptions.push(format!(
                    "Stop if {} has < {:.4} improvement for {} consecutive generations",
                    monitor, improvement_threshold, patience_generations
                ));
            }
            
            StoppingCriterion::TimeLimit { max_seconds } => {
                descriptions.push(format!(
                    "Stop after {} seconds ({:.1} minutes)",
                    max_seconds,
                    *max_seconds as f32 / 60.0
                ));
                // Assume ~2-5 seconds per generation as average
                estimated_generations.push(max_seconds / 3);
            }
            
            StoppingCriterion::TargetAccuracy { threshold } => {
                descriptions.push(format!(
                    "Stop when best accuracy reaches {:.1}%",
                    threshold * 100.0
                ));
            }
            
            StoppingCriterion::ManualStop => {
                descriptions.push("Can be stopped manually by user".to_string());
            }
        }
    }
    
    let policy_explanation = format!(
        "Evolution stops when {} of the criteria are triggered",
        match policy.to_lowercase().as_str() {
            "any" => "ANY",
            "all" => "ALL",
            _ => "UNKNOWN",
        }
    );
    
    Ok(StoppingPreview {
        estimated_generations,
        criterion_descriptions: descriptions,
        policy_explanation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generation_limit() {
        let mut state = EvolutionProgressState::new();
        let criteria = vec![StoppingCriterion::GenerationLimit { max_generations: 10 }];
        
        // Not yet reached
        state.on_generation_complete(5, 0.8, 0.9, 0.7, 0.5);
        let decision = check_stopping_criteria(&criteria, &state, "any", false);
        assert!(!decision.should_stop);
        
        // Reached limit
        state.on_generation_complete(10, 0.81, 0.91, 0.71, 0.5);
        let decision = check_stopping_criteria(&criteria, &state, "any", false);
        assert!(decision.should_stop);
        assert!(decision.triggered_criteria.contains(&"GenerationLimit".to_string()));
    }
    
    #[test]
    fn test_target_accuracy() {
        let mut state = EvolutionProgressState::new();
        let criteria = vec![StoppingCriterion::TargetAccuracy { threshold: 0.95 }];
        
        state.on_generation_complete(0, 0.8, 0.81, 0.7, 0.5);
        let decision = check_stopping_criteria(&criteria, &state, "any", false);
        assert!(!decision.should_stop);
        
        state.on_generation_complete(1, 0.85, 0.96, 0.75, 0.5);
        let decision = check_stopping_criteria(&criteria, &state, "any", false);
        assert!(decision.should_stop);
        assert!(decision.triggered_criteria.contains(&"TargetAccuracy".to_string()));
    }
    
    #[test]
    fn test_fitness_plateau_best_fitness() {
        let mut state = EvolutionProgressState::new();
        let criteria = vec![StoppingCriterion::FitnessPlateau {
            patience_generations: 3,
            improvement_threshold: 0.01,
            monitor: "best_fitness".to_string(),
        }];
        
        // Build up history without improvement
        state.on_generation_complete(0, 0.80, 0.80, 0.70, 0.5);
        state.on_generation_complete(1, 0.80, 0.80, 0.70, 0.5);
        state.on_generation_complete(2, 0.80, 0.80, 0.70, 0.5);
        state.on_generation_complete(3, 0.80, 0.80, 0.70, 0.5);
        
        let decision = check_stopping_criteria(&criteria, &state, "any", false);
        assert!(decision.should_stop);
        assert!(decision.triggered_criteria.contains(&"FitnessPlateau".to_string()));
    }
    
    #[test]
    fn test_manual_stop() {
        let state = EvolutionProgressState::new();
        let criteria = vec![StoppingCriterion::ManualStop];
        
        let decision = check_stopping_criteria(&criteria, &state, "any", false);
        assert!(!decision.should_stop);
        
        let decision = check_stopping_criteria(&criteria, &state, "any", true);
        assert!(decision.should_stop);
        assert!(decision.triggered_criteria.contains(&"ManualStop".to_string()));
    }
    
    #[test]
    fn test_policy_any() {
        let state = EvolutionProgressState::new();
        let _criteria = vec![
            StoppingCriterion::GenerationLimit { max_generations: 100 },
            StoppingCriterion::TargetAccuracy { threshold: 0.99 },
        ];
        
        // With "any" policy, if ManualStop is triggered, we should stop
        let decision = check_stopping_criteria(
            &[StoppingCriterion::ManualStop],
            &state,
            "any",
            true,
        );
        assert!(decision.should_stop);
    }
    
    #[test]
    fn test_policy_all() {
        let mut state = EvolutionProgressState::new();
        state.on_generation_complete(100, 0.9, 0.95, 0.85, 0.5);
        
        let criteria = vec![
            StoppingCriterion::GenerationLimit { max_generations: 100 },
            StoppingCriterion::TargetAccuracy { threshold: 0.95 },
        ];
        
        // With "all" policy, both must be triggered
        let decision = check_stopping_criteria(&criteria, &state, "all", false);
        assert!(decision.should_stop); // Both triggered
    }
    
    #[test]
    fn test_validate_stopping_config() {
        // Valid config
        let criteria = vec![StoppingCriterion::GenerationLimit { max_generations: 10 }];
        assert!(validate_stopping_config(&criteria, "any").is_ok());
        
        // Invalid policy
        assert!(validate_stopping_config(&criteria, "unknown").is_err());
        
        // Invalid generation limit
        let bad_criteria = vec![StoppingCriterion::GenerationLimit { max_generations: 0 }];
        assert!(validate_stopping_config(&bad_criteria, "any").is_err());
        
        // Invalid accuracy threshold
        let bad_criteria = vec![StoppingCriterion::TargetAccuracy { threshold: 1.5 }];
        assert!(validate_stopping_config(&bad_criteria, "any").is_err());
        
        // Empty criteria
        assert!(validate_stopping_config(&[], "any").is_err());
    }
    
    #[test]
    fn test_generate_stopping_preview() {
        let criteria = vec![
            StoppingCriterion::GenerationLimit { max_generations: 50 },
            StoppingCriterion::TargetAccuracy { threshold: 0.95 },
        ];
        
        let preview = generate_stopping_preview(&criteria, "any").unwrap();
        assert!(!preview.criterion_descriptions.is_empty());
        assert!(preview.policy_explanation.contains("ANY"));
    }
}
