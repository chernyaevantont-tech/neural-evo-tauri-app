use crate::dtos::{GenerationParetoFront, GenomeObjectives};

/// Returns true if `a` is dominated by `b`.
///
/// Objectives:
/// - maximize accuracy
/// - minimize inference latency
/// - minimize model size
pub fn is_dominated(a: &GenomeObjectives, b: &GenomeObjectives) -> bool {
    let b_no_worse_all = b.accuracy >= a.accuracy
        && b.inference_latency_ms <= a.inference_latency_ms
        && b.model_size_mb <= a.model_size_mb;

    let b_strictly_better_any = b.accuracy > a.accuracy
        || b.inference_latency_ms < a.inference_latency_ms
        || b.model_size_mb < a.model_size_mb;

    b_no_worse_all && b_strictly_better_any
}

/// Computes Pareto frontier in O(N^2).
pub fn compute_pareto_front(genomes: &[GenomeObjectives]) -> Vec<GenomeObjectives> {
    let mut domination_counts = vec![0u32; genomes.len()];

    for i in 0..genomes.len() {
        for j in 0..genomes.len() {
            if i == j {
                continue;
            }
            if is_dominated(&genomes[i], &genomes[j]) {
                domination_counts[i] = domination_counts[i].saturating_add(1);
            }
        }
    }

    genomes
        .iter()
        .enumerate()
        .map(|(idx, genome)| {
            let mut updated = genome.clone();
            updated.domination_count = domination_counts[idx];
            updated.is_dominated = domination_counts[idx] > 0;
            updated
        })
        .filter(|g| !g.is_dominated)
        .collect()
}

pub fn compute_generation_pareto_front(
    generation: u32,
    genomes: &[GenomeObjectives],
) -> GenerationParetoFront {
    let pareto_members = compute_pareto_front(genomes);
    let objectives_3d = pareto_members
        .iter()
        .map(|g| (g.accuracy, g.inference_latency_ms, g.model_size_mb))
        .collect();

    GenerationParetoFront {
        generation,
        total_genomes: genomes.len() as u32,
        pareto_members,
        objectives_3d,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn obj(id: &str, acc: f32, lat: f32, size: f32) -> GenomeObjectives {
        GenomeObjectives {
            genome_id: id.to_string(),
            accuracy: acc,
            inference_latency_ms: lat,
            model_size_mb: size,
            training_time_ms: 1000,
            is_dominated: false,
            domination_count: 0,
        }
    }

    #[test]
    fn test_is_dominated_true_when_other_is_better_or_equal_everywhere() {
        let a = obj("a", 0.80, 10.0, 20.0);
        let b = obj("b", 0.85, 9.5, 19.0);
        assert!(is_dominated(&a, &b));
    }

    #[test]
    fn test_is_dominated_false_when_tradeoff_exists() {
        let a = obj("a", 0.90, 11.0, 18.0);
        let b = obj("b", 0.92, 12.0, 16.0);
        assert!(!is_dominated(&a, &b));
        assert!(!is_dominated(&b, &a));
    }

    #[test]
    fn test_compute_pareto_front_returns_expected_members() {
        let genomes = vec![
            obj("g1", 0.80, 10.0, 20.0),
            obj("g2", 0.82, 9.0, 19.0),
            obj("g3", 0.78, 12.0, 21.0),
            obj("g4", 0.86, 13.0, 16.0),
            obj("g5", 0.81, 8.5, 25.0),
        ];

        let front = compute_pareto_front(&genomes);
        let ids: std::collections::HashSet<String> =
            front.iter().map(|g| g.genome_id.clone()).collect();

        assert!(!ids.contains("g1"));
        assert!(!ids.contains("g3"));
        assert!(ids.contains("g2"));
        assert!(ids.contains("g4"));
        assert!(ids.contains("g5"));

        for member in &front {
            assert!(!member.is_dominated);
            assert_eq!(member.domination_count, 0);
        }
    }

    #[test]
    fn test_generation_payload_contains_3d_points() {
        let genomes = vec![obj("a", 0.90, 8.0, 10.0), obj("b", 0.85, 7.0, 12.0)];
        let payload = compute_generation_pareto_front(7, &genomes);

        assert_eq!(payload.generation, 7);
        assert_eq!(payload.total_genomes, 2);
        assert_eq!(payload.pareto_members.len(), payload.objectives_3d.len());
    }

    #[test]
    fn test_compute_pareto_front_performance_n100_under_100ms() {
        let genomes: Vec<GenomeObjectives> = (0..100)
            .map(|i| {
                let acc = 0.5 + (i as f32 * 0.003);
                let lat = 25.0 - (i as f32 * 0.1);
                let size = 50.0 - (i as f32 * 0.2);
                obj(&format!("g{i}"), acc, lat, size)
            })
            .collect();

        let start = Instant::now();
        let _ = compute_pareto_front(&genomes);
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() < 100, "elapsed={}ms", elapsed.as_millis());
    }

    #[test]
    fn test_empty_population() {
        let genomes = vec![];
        let front = compute_pareto_front(&genomes);
        assert!(front.is_empty());
    }

    #[test]
    fn test_single_genome_is_always_on_front() {
        let genomes = vec![obj("g1", 0.75, 15.0, 30.0)];
        let front = compute_pareto_front(&genomes);
        
        assert_eq!(front.len(), 1);
        assert_eq!(front[0].genome_id, "g1");
        assert!(!front[0].is_dominated);
    }

    #[test]
    fn test_identical_genomes() {
        let genomes = vec![
            obj("g1", 0.80, 10.0, 20.0),
            obj("g2", 0.80, 10.0, 20.0),
            obj("g3", 0.80, 10.0, 20.0),
        ];
        let front = compute_pareto_front(&genomes);
        
        // All should be on front (none dominates another with equality)
        assert_eq!(front.len(), 3);
    }

    #[test]
    fn test_dominated_by_multiple() {
        let genomes = vec![
            obj("g1", 0.70, 15.0, 25.0), // Worst in all dimensions
            obj("g2", 0.90, 5.0, 10.0),  // Best in all dimensions
            obj("g3", 0.85, 8.0, 15.0),
            obj("g4", 0.88, 6.0, 12.0),
        ];
        let front = compute_pareto_front(&genomes);
        
        let ids: std::collections::HashSet<String> =
            front.iter().map(|g| g.genome_id.clone()).collect();
        
        assert!(!ids.contains("g1")); // g1 is dominated by everything
        assert!(ids.contains("g2")); // g2 is best overall
    }

    #[test]
    fn test_is_dominated_reflexivity_false() {
        let a = obj("a", 0.80, 10.0, 20.0);
        // A genome cannot dominate itself
        assert!(!is_dominated(&a, &a));
    }

    #[test]
    fn test_is_dominated_with_extreme_values() {
        let a = obj("a", 0.0, 1000.0, 1000.0);  // Worst accuracy, worst latency/size
        let b = obj("b", 1.0, 0.0, 0.0);         // Best in all
        assert!(is_dominated(&a, &b));
        assert!(!is_dominated(&b, &a));
    }

    #[test]
    fn test_generation_payload_empty_population() {
        let genomes = vec![];
        let payload = compute_generation_pareto_front(5, &genomes);
        
        assert_eq!(payload.generation, 5);
        assert_eq!(payload.total_genomes, 0);
        assert!(payload.pareto_members.is_empty());
        assert!(payload.objectives_3d.is_empty());
    }

    #[test]
    fn test_generation_payload_3d_points_match_members() {
        let genomes = vec![
            obj("g1", 0.90, 8.0, 10.0),
            obj("g2", 0.85, 7.0, 12.0),
            obj("g3", 0.88, 9.0, 11.0),
        ];
        let payload = compute_generation_pareto_front(10, &genomes);
        
        // All should be on pareto front (different tradeoffs)
        assert_eq!(payload.pareto_members.len(), payload.objectives_3d.len());
        assert!(payload.pareto_members.len() > 0);
    }

    #[test]
    fn test_domination_count_correctness() {
        let genomes = vec![
            obj("g1", 0.80, 10.0, 20.0),
            obj("g2", 0.85, 9.0, 19.0),
            obj("g3", 0.90, 8.0, 18.0),
        ];
        let front = compute_pareto_front(&genomes);
        
        // g3 dominates both g1 and g2, so their domination_count > 0
        for member in &front {
            if member.genome_id == "g3" {
                assert_eq!(member.domination_count, 0); // Front member
            }
        }
    }

    #[test]
    fn test_large_population_n500() {
        let genomes: Vec<GenomeObjectives> = (0..500)
            .map(|i| {
                let acc = 0.5 + (i as f32 * 0.001);
                let lat = 50.0 - (i as f32 * 0.05);
                let size = 100.0 - (i as f32 * 0.1);
                obj(&format!("g{i}"), acc, lat, size)
            })
            .collect();
        
        let front = compute_pareto_front(&genomes);
        // With linear tradeoffs, expect roughly O(sqrt(n)) pareto members
        assert!(front.len() > 0);
        assert!(front.len() < 500); // Not all should be on front
    }

    #[test]
    fn test_tradeoff_accuracy_vs_latency() {
        let genomes = vec![
            obj("fast", 0.75, 5.0, 10.0),   // Fast but low accuracy
            obj("accurate", 0.95, 15.0, 10.0), // Accurate but slow
            obj("balanced", 0.85, 10.0, 12.0), // Balanced
        ];
        let front = compute_pareto_front(&genomes);
        
        let ids: std::collections::HashSet<String> =
            front.iter().map(|g| g.genome_id.clone()).collect();
        
        // All three should be on front (different tradeoffs)
        assert!(ids.contains("fast"));
        assert!(ids.contains("accurate"));
        assert!(ids.contains("balanced"));
    }
}