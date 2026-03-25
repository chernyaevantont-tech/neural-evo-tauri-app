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
}