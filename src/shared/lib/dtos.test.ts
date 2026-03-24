import { describe, expect, it } from 'vitest';
import type { GenomeObjectives } from './dtos';

describe('DTOs', () => {
    it('creates GenomeObjectives with typed fields', () => {
        const obj: GenomeObjectives = {
            genome_id: 'g1',
            accuracy: 0.95,
            inference_latency_ms: 50,
            model_size_mb: 10,
            training_time_ms: 5000,
            is_dominated: false,
            domination_count: 0,
        };

        expect(obj.accuracy).toBe(0.95);
    });
});
