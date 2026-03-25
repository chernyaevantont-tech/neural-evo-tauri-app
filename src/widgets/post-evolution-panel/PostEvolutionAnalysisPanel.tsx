import React, { useState } from 'react';
import type { GenerationParetoFront, GenomeGenealogy } from '../../shared/lib';
import { GenealogicTreeView } from '../genealogy-tree-viewer';

type Props = {
    genealogyTree?: Map<string, GenomeGenealogy>;
    paretoHistory: Map<number, GenerationParetoFront>;
    onSyncGenealogyTree?: (tree: Map<string, GenomeGenealogy>) => void;
};

export function PostEvolutionAnalysisPanel({ genealogyTree, paretoHistory, onSyncGenealogyTree }: Props) {
    const [tab, setTab] = useState<'summary' | 'genealogy'>('summary');

    return (
        <section>
            <div style={{ display: 'inline-flex', border: '1px solid var(--color-border-primary)', borderRadius: 8, overflow: 'hidden', marginBottom: 12 }}>
                <button type="button" onClick={() => setTab('summary')} style={{ padding: '6px 12px', border: 'none', background: tab === 'summary' ? 'var(--color-accent-primary, #646cff)' : 'transparent', color: tab === 'summary' ? '#fff' : 'var(--color-text-secondary)' }}>Summary</button>
                <button type="button" onClick={() => setTab('genealogy')} style={{ padding: '6px 12px', border: 'none', background: tab === 'genealogy' ? 'var(--color-accent-primary, #646cff)' : 'transparent', color: tab === 'genealogy' ? '#fff' : 'var(--color-text-secondary)' }}>Genealogy</button>
            </div>

            {tab === 'summary' ? (
                <div style={{ color: 'var(--color-text-secondary)' }}>Post-evolution summary panel.</div>
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
