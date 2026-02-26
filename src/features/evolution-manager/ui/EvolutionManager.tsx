import React from 'react';
import styles from './EvolutionManager.module.css';
import { useEvolutionSettingsStore, CrossoverStrategy } from '../model/store';
import { useCanvasGenomeStore } from '../../../entities/canvas-genome';
import { useCanvasStateStore } from '../../../entities/canvas-state';

export const EvolutionManager: React.FC = () => {
    const settings = useEvolutionSettingsStore();
    const canvasNodesCount = useCanvasGenomeStore(state => state.nodes.size);

    const handleCrossoverChange = (strategy: CrossoverStrategy) => {
        settings.toggleCrossover(strategy);
    };

    const handleRateChange = (key: keyof typeof settings.mutationRates, e: React.ChangeEvent<HTMLInputElement>) => {
        settings.setMutationRate(key, parseFloat(e.target.value));
    };

    const handleMutateCurrent = () => {
        const genomesMap = useCanvasGenomeStore.getState().genomes;
        const genomesArray = Array.from(genomesMap.values());

        if (genomesArray.length === 0) {
            alert("Need at least 1 genome on the canvas to mutate!");
            return;
        }

        // Randomly pick one genome to mutate
        const targetIdx = Math.floor(Math.random() * genomesArray.length);
        const targetGenomeEntry = genomesArray[targetIdx];
        const targetGenome = targetGenomeEntry.genome;
        let mutatedResult = null;

        const nodesInGenomeCount = useCanvasGenomeStore.getState().genomeNode.get(targetGenomeEntry.id)?.length || "?";
        console.log(`[Mutate Current] Selected genome ${targetGenomeEntry.id} with ${nodesInGenomeCount} nodes to undergo mutation.`);

        // Roll probabilities
        const rRemove = Math.random();
        const rAdd = Math.random();

        if (rRemove < settings.mutationRates.removeNode) {
            console.log(`[Mutate Current] Rolled ${rRemove.toFixed(2)} < ${settings.mutationRates.removeNode}. Firing RemoveNode mutation...`);
            try {
                mutatedResult = targetGenome.MutateRemoveNode();
            } catch (e) {
                console.error("Remove Node Mutation failed:", e);
            }
        }

        // If remove didn't trigger or failed, maybe add node triggers
        if (!mutatedResult && rAdd < settings.mutationRates.addNode) {
            console.log(`[Mutate Current] Rolled ${rAdd.toFixed(2)} < ${settings.mutationRates.addNode}. Firing AddNode mutation...`);
            try {
                mutatedResult = targetGenome.MutateAddNode();
            } catch (e) {
                console.error("Add Node Mutation failed:", e);
            }
        }

        // TODO: add bypass connections & type changes later!

        if (mutatedResult) {
            console.log(`[Mutate Current] Mutation successful. Spawning new genome descendant!`);
            const store = useCanvasGenomeStore.getState();
            const canvasState = useCanvasStateStore.getState();
            store.addGenome(
                mutatedResult.nodes,
                mutatedResult.genome,
                canvasState.canvasWidth,
                canvasState.canvasHeight,
                canvasState.translate.x,
                canvasState.translate.y,
                canvasState.scale,
                100
            );
        } else {
            console.log("No structural mutations triggered or valid.");
        }
    };

    const handleBreed = () => {
        const genomesMap = useCanvasGenomeStore.getState().genomes;
        const genomesArray = Array.from(genomesMap.values());

        if (genomesArray.length < 2) {
            alert("Need at least 2 genomes on the canvas to breed!");
            return;
        }

        // Randomly pick one genome as recipient, and another as donor
        const recipientIdx = Math.floor(Math.random() * genomesArray.length);
        let donorIdx = Math.floor(Math.random() * genomesArray.length);
        while (donorIdx === recipientIdx) {
            donorIdx = Math.floor(Math.random() * genomesArray.length);
        }

        const recipient = genomesArray[recipientIdx].genome;
        const donor = genomesArray[donorIdx].genome;

        const maxNodes = settings.useMaxNodesLimit ? settings.maxNodesLimit : undefined;

        // Try breed 10 times to find valid insertion
        let result = null;
        for (let i = 0; i < 10; i++) {
            try {
                // Assuming Subgraph Insertion is selected since it's the only one inside Genome.Breed currently
                let breedResult = recipient.Breed(donor, maxNodes);
                if (breedResult) {
                    result = breedResult;
                    break;
                }
            } catch (e) {
                console.error(e);
            }
        }

        if (result) {
            const store = useCanvasGenomeStore.getState();
            const canvasState = useCanvasStateStore.getState();

            store.addGenome(
                result.nodes,
                result.genome,
                canvasState.canvasWidth,
                canvasState.canvasHeight,
                canvasState.translate.x,
                canvasState.translate.y,
                canvasState.scale,
                100 // layout iterations
            );
        } else {
            alert("Breed failed after 10 attempts. (Maybe exceeded node limit or no valid insertion points)");
        }
    };

    return (
        <div className={styles.container}>
            <h3 className={styles.title}>Evolution Manager</h3>

            <div className={styles.section}>
                <h4 className={styles.sectionTitle}>Crossover Strategies</h4>
                {(['subgraph-insertion', 'subgraph-replacement', 'neat-style', 'multi-point'] as CrossoverStrategy[]).map(strategy => (
                    <label key={strategy} className={styles.label}>
                        <input
                            type="checkbox"
                            checked={settings.selectedCrossovers.includes(strategy)}
                            onChange={() => handleCrossoverChange(strategy)}
                        />
                        {strategy.replace('-', ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </label>
                ))}
            </div>

            <div className={styles.section}>
                <h4 className={styles.sectionTitle}>Mutation Probabilities</h4>
                {Object.entries(settings.mutationRates).map(([key, value]) => (
                    <div key={key} className={styles.row}>
                        <span className={styles.label}>{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}</span>
                        <div style={{ display: 'flex', alignItems: 'center' }}>
                            <input
                                type="range"
                                min="0" max="1" step="0.05"
                                value={value}
                                onChange={(e) => handleRateChange(key as any, e)}
                                className={styles.slider}
                            />
                            <span className={styles.sliderValue}>{Math.round(value * 100)}%</span>
                        </div>
                    </div>
                ))}
            </div>

            <div className={styles.section}>
                <h4 className={styles.sectionTitle}>Bloat Control</h4>

                <label className={styles.label}>
                    <input type="checkbox" checked={settings.useMaxNodesLimit} onChange={e => settings.setUseMaxNodesLimit(e.target.checked)} />
                    Global Node Limit
                    {settings.useMaxNodesLimit && (
                        <input type="number" className={styles.input} value={settings.maxNodesLimit} onChange={e => settings.setMaxNodesLimit(parseInt(e.target.value) || 0)} />
                    )}
                </label>

                <label className={styles.label}>
                    <input type="checkbox" checked={settings.useParsimonyPressure} onChange={e => settings.setUseParsimonyPressure(e.target.checked)} />
                    Parsimony Pressure (α)
                    {settings.useParsimonyPressure && (
                        <input type="number" step="0.001" className={styles.input} style={{ width: 60 }} value={settings.parsimonyAlpha} onChange={e => settings.setParsimonyAlpha(parseFloat(e.target.value) || 0)} />
                    )}
                </label>

                <label className={styles.label}>
                    <input type="checkbox" checked={settings.useResourceAwareFitness} onChange={e => settings.setUseResourceAwareFitness(e.target.checked)} />
                    Resource-Aware Fitness
                </label>
            </div>

            <div className={styles.section}>
                <h4 className={styles.sectionTitle}>Actions</h4>
                <div className={styles.buttonGroup}>
                    <button className={styles.button} onClick={handleMutateCurrent}>Mutate Current</button>
                    <button className={styles.button} onClick={handleBreed}>Breed Random</button>
                </div>
                <div className={styles.metrics}>
                    <span>Current Nodes on Canvas: {canvasNodesCount}</span>
                </div>
            </div>
        </div>
    );
};
