import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import type { GenerationParetoFront, GenomeObjectives } from '../../shared/lib';
import { ParetoFrontVisualizer } from './ParetoFrontVisualizer';
import { ParetoScatterPlot } from './ParetoScatterPlot';
import { ParetoSelector } from './ParetoSelector';

let capturedScatterProps: any;

vi.mock('react-chartjs-2', () => ({
    Scatter: (props: any) => {
        capturedScatterProps = props;
        return <div data-testid="scatter-mock" />;
    },
}));

const allGenomes: GenomeObjectives[] = [
    {
        genome_id: 'g-front',
        accuracy: 0.95,
        inference_latency_ms: 8,
        model_size_mb: 2,
        training_time_ms: 1400,
        is_dominated: false,
        domination_count: 0,
    },
    {
        genome_id: 'g-dom',
        accuracy: 0.87,
        inference_latency_ms: 14,
        model_size_mb: 3,
        training_time_ms: 1300,
        is_dominated: true,
        domination_count: 2,
        device_feasible: false,
    },
];

const pareto: GenerationParetoFront = {
    generation: 4,
    total_genomes: 2,
    pareto_members: [allGenomes[0]],
    objectives_3d: [[0.95, 8, 2]],
    all_genomes: allGenomes,
    frontier_genome_ids: ['g-front'],
};

describe('Pareto front visualizer parts', () => {
    beforeEach(() => {
        capturedScatterProps = undefined;
    });

    it('renders frontier and dominated datasets separately', () => {
        render(
            <ParetoScatterPlot
                pareto={pareto}
                onSelectGenome={() => {}}
            />,
        );

        expect(screen.getByTestId('scatter-mock')).toBeTruthy();
        expect(capturedScatterProps.data.datasets).toHaveLength(2);
        expect(capturedScatterProps.data.datasets[0].data).toHaveLength(1);
        expect(capturedScatterProps.data.datasets[1].data).toHaveLength(1);
    });

    it('calls onSelectGenome when point is clicked', () => {
        const onSelect = vi.fn();
        render(
            <ParetoScatterPlot
                pareto={pareto}
                onSelectGenome={onSelect}
            />,
        );

        const chartLike = {
            data: capturedScatterProps.data,
        };

        capturedScatterProps.options.onClick(
            {} as any,
            [{ datasetIndex: 1, index: 0 }],
            chartLike as any,
        );

        expect(onSelect).toHaveBeenCalledWith('g-front');
    });

    it('shows tooltip fields including feasibility', () => {
        render(
            <ParetoScatterPlot
                pareto={pareto}
                onSelectGenome={() => {}}
            />,
        );

        const tooltipLabel = capturedScatterProps.options.plugins.tooltip.callbacks.label;
        const tooltipText = tooltipLabel({ raw: capturedScatterProps.data.datasets[0].data[0] }).join(' | ');

        expect(tooltipText).toContain('Genome: g-dom');
        expect(tooltipText).toContain('Accuracy: 0.8700');
        expect(tooltipText).toContain('Latency: 14.000 ms');
        expect(tooltipText).toContain('Device feasible: no');
    });

    it('applies selected point outline style', () => {
        render(
            <ParetoScatterPlot
                pareto={pareto}
                selectedGenomeId="g-front"
                onSelectGenome={() => {}}
            />,
        );

        const pointBorderWidth = capturedScatterProps.data.datasets[1].pointBorderWidth;
        const width = pointBorderWidth({ raw: { genomeId: 'g-front' } });
        expect(width).toBe(3);
    });

    it('selector actions invoke handlers for selected genome', () => {
        const onUseAsSeed = vi.fn();
        const onOpenDetails = vi.fn();
        const onExportSelected = vi.fn();

        render(
            <ParetoSelector
                selectedGenome={allGenomes[0]}
                onUseAsSeed={onUseAsSeed}
                onOpenDetails={onOpenDetails}
                onExportSelected={onExportSelected}
            />,
        );

        screen.getByRole('button', { name: 'Use as seed' }).click();
        screen.getByRole('button', { name: 'Open details' }).click();
        screen.getByRole('button', { name: 'Export selected' }).click();

        expect(onUseAsSeed).toHaveBeenCalledWith('g-front');
        expect(onOpenDetails).toHaveBeenCalledWith('g-front');
        expect(onExportSelected).toHaveBeenCalledWith('g-front');
    });

    it('passes feasibility maps and filters to feasible-only points', () => {
        render(
            <ParetoFrontVisualizer
                currentParetoFront={pareto.pareto_members}
                paretoHistory={new Map([[pareto.generation, pareto]])}
                feasibilityByGenomeId={{ 'g-front': true, 'g-dom': false }}
                constraintViolationScoreByGenomeId={{ 'g-front': 0, 'g-dom': 0.8 }}
                showOnlyFeasible={true}
            />,
        );

        const dominatedPoints = capturedScatterProps.data.datasets[0].data;
        const frontierPoints = capturedScatterProps.data.datasets[1].data;

        expect(dominatedPoints).toHaveLength(0);
        expect(frontierPoints).toHaveLength(1);
        expect(frontierPoints[0].genomeId).toBe('g-front');
    });
});