import React from 'react';
import styles from './GenealogicTreeView.module.css';
import type { GenealogyGraphFilters } from '../../shared/hooks/useGenealogyGraph';

type Props = {
    filters: GenealogyGraphFilters;
    generationBounds: { min: number; max: number };
    fitnessBounds: { min: number; max: number };
    onChange: (next: GenealogyGraphFilters) => void;
};

export function GenealogyFilters({
    filters,
    generationBounds,
    fitnessBounds,
    onChange,
}: Props) {
    const update = (patch: Partial<GenealogyGraphFilters>) => {
        onChange({ ...filters, ...patch });
    };

    return (
        <div className={styles.filtersPanel}>
            <label className={styles.filterItem}>
                Gen min
                <input
                    aria-label="Generation min"
                    type="number"
                    min={generationBounds.min}
                    max={generationBounds.max}
                    value={filters.generationMin}
                    onChange={(event) => update({ generationMin: Number(event.target.value) })}
                />
            </label>

            <label className={styles.filterItem}>
                Gen max
                <input
                    aria-label="Generation max"
                    type="number"
                    min={generationBounds.min}
                    max={generationBounds.max}
                    value={filters.generationMax}
                    onChange={(event) => update({ generationMax: Number(event.target.value) })}
                />
            </label>

            <label className={styles.filterItem}>
                Fitness min
                <input
                    aria-label="Fitness min"
                    type="number"
                    step="0.0001"
                    min={fitnessBounds.min}
                    max={fitnessBounds.max}
                    value={filters.fitnessMin}
                    onChange={(event) => update({ fitnessMin: Number(event.target.value) })}
                />
            </label>

            <label className={styles.filterItem}>
                Fitness max
                <input
                    aria-label="Fitness max"
                    type="number"
                    step="0.0001"
                    min={fitnessBounds.min}
                    max={fitnessBounds.max}
                    value={filters.fitnessMax}
                    onChange={(event) => update({ fitnessMax: Number(event.target.value) })}
                />
            </label>

            <label className={styles.filterCheckbox}>
                <input
                    aria-label="Pareto only"
                    type="checkbox"
                    checked={filters.paretoOnly}
                    onChange={(event) => update({ paretoOnly: event.target.checked })}
                />
                Pareto only
            </label>

            <label className={styles.filterItem}>
                Ancestors depth
                <input
                    aria-label="Ancestors depth"
                    type="number"
                    min={0}
                    max={10}
                    value={filters.ancestorsDepth}
                    onChange={(event) => update({ ancestorsDepth: Math.max(0, Number(event.target.value) || 0) })}
                />
            </label>
        </div>
    );
}
