use neural_evo_tauri_app_lib::device_profiles::{
    score_fitness_with_device_constraints, DeviceResourceConstraints,
};
use neural_evo_tauri_app_lib::dtos::{GenomeObjectives, MutationType, StoppingCriterion};
use neural_evo_tauri_app_lib::genealogy::GenealogyStore;
use neural_evo_tauri_app_lib::pareto;
use neural_evo_tauri_app_lib::profiler::ProfilerCollector;
use neural_evo_tauri_app_lib::stopping_criteria::{
    check_stopping_criteria, EvolutionProgressState,
};

fn obj(id: &str, acc: f32, latency: f32, size: f32) -> GenomeObjectives {
    GenomeObjectives {
        genome_id: id.to_string(),
        accuracy: acc,
        inference_latency_ms: latency,
        model_size_mb: size,
        training_time_ms: 1200,
        is_dominated: false,
        domination_count: 0,
    }
}

#[test]
fn integration_multi_objective_device_and_profiler_pipeline() {
    let constraints = DeviceResourceConstraints {
        mops_budget: 200.0,
        ram_budget_mb: 16.0,
        flash_budget_mb: 6.0,
        max_latency_ms: 12.0,
    };

    let genomes = vec![
        obj("g-fast", 0.91, 6.0, 1.5),
        obj("g-large", 0.94, 10.0, 9.0),
        obj("g-slow", 0.89, 30.0, 4.0),
    ];

    let mut adjusted = Vec::new();
    let mut feasibility = Vec::new();

    for g in &genomes {
        let (adjusted_fitness, validation) =
            score_fitness_with_device_constraints(g.accuracy, g, &constraints, 0.5);
        let mut updated = g.clone();
        updated.accuracy = adjusted_fitness;
        adjusted.push(updated);
        feasibility.push(validation.is_feasible);
    }

    let front = pareto::compute_generation_pareto_front(7, &adjusted);

    assert_eq!(front.generation, 7);
    assert_eq!(front.total_genomes, genomes.len() as u32);
    assert_eq!(front.pareto_members.len(), front.objectives_3d.len());
    assert!(feasibility.iter().any(|f| !f), "expected at least one infeasible genome");

    let mut profiler = ProfilerCollector::new();
    profiler.mark_train_start();
    profiler.record_batch(32);
    profiler.record_batch(32);
    profiler.mark_first_batch();
    profiler.set_model_params_mb(12.0);
    profiler.set_gradients_mb(8.0);
    profiler.set_optimizer_state_mb(7.0);
    profiler.set_activation_mb(10.0);
    profiler.mark_train_end();
    profiler.mark_val_start();
    profiler.record_inference_samples(64);
    profiler.mark_val_end();
    profiler.mark_test_start();
    profiler.mark_test_end();

    let summary = profiler.finalize();
    assert_eq!(summary.batch_count, 2);
    assert!(summary.peak_active_memory_mb >= 37.0);
    assert!(summary.total_train_duration_ms <= summary.train_end_ms.saturating_sub(summary.train_start_ms));
}

#[test]
fn integration_genealogy_and_stopping_reason_propagation() {
    let mut genealogy = GenealogyStore::new();

    genealogy
        .register_founder("g-parent-a".to_string(), 0)
        .expect("register founder a");
    genealogy
        .register_founder("g-parent-b".to_string(), 0)
        .expect("register founder b");
    genealogy
        .register_crossover(
            "g-parent-a".to_string(),
            "g-parent-b".to_string(),
            "g-child".to_string(),
            1,
        )
        .expect("register crossover");
    genealogy
        .register_mutation(
            "g-child".to_string(),
            "g-grandchild".to_string(),
            MutationType::AddNode {
                node_type: "Dense".to_string(),
                source: "in".to_string(),
                target: "out".to_string(),
            },
            2,
        )
        .expect("register mutation");

    let path = genealogy
        .get_genealogy("g-grandchild")
        .expect("genealogy path exists");
    assert_eq!(path.target_genome_id, "g-grandchild");
    assert_eq!(path.records.len(), 4);
    assert_eq!(path.edges.len(), 3);

    let mut state = EvolutionProgressState::new();
    state.on_generation_complete(8, 0.93, 0.95, 0.91, 0.5);

    let criteria = vec![
        StoppingCriterion::GenerationLimit {
            max_generations: 5,
        },
        StoppingCriterion::TargetAccuracy { threshold: 0.9 },
    ];

    let decision = check_stopping_criteria(&criteria, &state, "any", false);
    assert!(decision.should_stop);
    assert!(decision
        .triggered_criteria
        .iter()
        .any(|c| c == "GenerationLimit"));
    assert!(decision
        .reason_message
        .contains("Generation limit reached")
        || decision.reason_message.contains("Target accuracy reached"));
}
