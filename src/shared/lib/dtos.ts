export interface TrainingProfiler {
    train_start_ms: number;
    first_batch_ms: number;
    train_end_ms: number;
    total_train_duration_ms: number;
    val_start_ms: number;
    val_end_ms: number;
    val_duration_ms: number;
    test_start_ms: number;
    test_end_ms: number;
    test_duration_ms: number;
    peak_active_memory_mb: number;
    peak_model_params_mb: number;
    peak_gradient_mb: number;
    peak_optim_state_mb: number;
    peak_activation_mb: number;
    samples_per_sec: number;
    inference_msec_per_sample: number;
    batch_count: number;
    early_stop_epoch?: number;
}

export interface GenomeObjectives {
    genome_id: string;
    accuracy: number;
    inference_latency_ms: number;
    model_size_mb: number;
    training_time_ms: number;
    train_time_ms?: number;
    is_dominated: boolean;
    domination_count: number;
    device_feasible?: boolean;
    constraint_violation_score?: number;
}

export interface GenerationParetoFront {
    generation: number;
    total_genomes: number;
    pareto_members: GenomeObjectives[];
    objectives_3d: [number, number, number][];
    all_genomes?: GenomeObjectives[];
    frontier_genome_ids?: string[];
}

export type ComputeType = 'ARM' | 'X86' | 'GPU';

export interface DeviceProfile {
    device_id: string;
    device_name: string;
    compute_capability: ComputeType;
    ram_mb: number;
    vram_mb?: number;
    inference_latency_budget_ms: number;
    training_available: boolean;
    power_budget_mw?: number;
    max_model_size_mb?: number;
    target_fps?: number;
}

export interface DeviceResourceConstraints {
    mops_budget: number;
    ram_budget_mb: number;
    flash_budget_mb: number;
    max_latency_ms: number;
}

export interface DeviceTemplateDto {
    id: string;
    name: string;
    constraints: DeviceResourceConstraints;
    notes?: string;
    tags: string[];
    created_at_unix_ms: number;
    updated_at_unix_ms: number;
}

export interface CreateDeviceTemplateInput {
    name: string;
    constraints: DeviceResourceConstraints;
    notes?: string;
    tags: string[];
}

export interface UpdateDeviceTemplatePatch {
    name?: string;
    constraints?: DeviceResourceConstraints;
    notes?: string;
    tags?: string[];
}

export type DeviceLibraryImportMode = 'merge' | 'replace';

export type MutationType =
    | { type: 'Random' }
    | { type: 'AddNode'; data: { node_type: string; source: string; target: string } }
    | { type: 'RemoveNode'; data: { node_id: string } }
    | { type: 'RemoveSubgraph'; data: { node_ids: string[] } }
    | { type: 'ParameterMutation'; data: { layer_id: string; param_name: string } }
    | { type: 'ParameterScale'; data: { layer_id: string; scale_factor: number } }
    | { type: 'Crossover'; data: { parent1: string; parent2: string } };

export interface GenomeGenealogy {
    genome_id: string;
    generation: number;
    parent_ids: string[];
    mutation_type: MutationType;
    mutation_params: Record<string, unknown>;
    fitness: number;
    accuracy: number;
    created_at_ms: number;
}

export type StoppingCriterionType =
    | { type: 'GenerationLimit'; max_generations: number }
    | {
        type: 'FitnessPlateau';
        patience_generations: number;
        improvement_threshold: number;
        monitor: 'best_fitness' | 'pareto_coverage' | 'population_avg';
    }
    | { type: 'TimeLimit'; max_seconds: number }
    | { type: 'TargetAccuracy'; threshold: number }
    | { type: 'ManualStop' };

export interface StoppingPolicy {
    criteria: StoppingCriterionType[];
    policy_type: 'any' | 'all';
}
