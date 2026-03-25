import React, { useState } from 'react';
import type { GenerationParetoFront, GenomeGenealogy } from '../../shared/lib';
import { ParetoFrontVisualizer } from '../pareto-front-visualizer';
import { GenealogicTreeView } from '../genealogy-tree-viewer';

type Props = {
    genealogyTree?: Map<string, GenomeGenealogy>;
    paretoHistory: Map<number, GenerationParetoFront>;
    onSyncGenealogyTree?: (tree: Map<string, GenomeGenealogy>) => void;
};

export function EvolutionDashboardTabs({ genealogyTree, paretoHistory, onSyncGenealogyTree }: Props) {
    const [tab, setTab] = useState<'pareto' | 'genealogy'>('pareto');

    return (
        <section>
            <div style={{ display: 'inline-flex', border: '1px solid var(--color-border-primary)', borderRadius: 8, overflow: 'hidden', marginBottom: 12 }}>
                <button type="button" onClick={() => setTab('pareto')} style={{ padding: '6px 12px', border: 'none', background: tab === 'pareto' ? 'var(--color-accent-primary, #646cff)' : 'transparent', color: tab === 'pareto' ? '#fff' : 'var(--color-text-secondary)' }}>Pareto</button>
                <button type="button" onClick={() => setTab('genealogy')} style={{ padding: '6px 12px', border: 'none', background: tab === 'genealogy' ? 'var(--color-accent-primary, #646cff)' : 'transparent', color: tab === 'genealogy' ? '#fff' : 'var(--color-text-secondary)' }}>Genealogy</button>
            </div>

            {tab === 'pareto' ? (
                <ParetoFrontVisualizer currentParetoFront={[]} paretoHistory={paretoHistory} />
            ) : (
                <GenealogicTreeView
                    genealogyTree={genealogyTree}
                    paretoHistory={paretoHistory}
                    onGenealogyTreeSync={onSyncGenealogyTree}
                />
            )}
        </section>
    );
}
