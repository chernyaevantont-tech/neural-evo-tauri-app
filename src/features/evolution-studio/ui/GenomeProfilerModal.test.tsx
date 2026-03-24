import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { GenomeProfilerModal } from './GenomeProfilerModal';

describe('GenomeProfilerModal', () => {
    it('displays profiler sections and values', () => {
        const profiler = {
            train_start_ms: 0,
            first_batch_ms: 100,
            train_end_ms: 5000,
            total_train_duration_ms: 5000,
            val_start_ms: 5001,
            val_end_ms: 6000,
            val_duration_ms: 999,
            test_start_ms: 6001,
            test_end_ms: 7000,
            test_duration_ms: 999,
            peak_active_memory_mb: 128.4,
            peak_model_params_mb: 12.1,
            peak_gradient_mb: 13.2,
            peak_optim_state_mb: 14.3,
            peak_activation_mb: 88.8,
            samples_per_sec: 42.5,
            inference_msec_per_sample: 0.456,
            batch_count: 100,
            early_stop_epoch: 3,
        };

        render(
            <GenomeProfilerModal
                genomeId="g1"
                profiler={profiler}
                onClose={() => {}}
            />,
        );

        expect(screen.getByText(/Profiler: g1/)).toBeTruthy();
        expect(screen.getByText(/Training/)).toBeTruthy();
        expect(screen.getByText(/Validation & Test/)).toBeTruthy();
        expect(screen.getByText(/Memory Peaks/)).toBeTruthy();
        expect(screen.getByText(/5.00s/)).toBeTruthy();
    });
});
