import React from 'react';
import styles from './EvolutionManager.module.css';
import { useEvolutionSettingsStore, CrossoverStrategy } from '../model/store';
import { useCanvasGenomeStore } from '../../../entities/canvas-genome';
import { useCanvasStateStore } from '../../../entities/canvas-state';

export const EvolutionManager: React.FC = () => {
    const settings = useEvolutionSettingsStore();
    const canvasNodesCount = useCanvasGenomeStore(state => state.nodes.size);

    const selectedGenomeIdRaw = useCanvasStateStore(state => state.selectedGenomeId);
    const selectedNodeId = useCanvasStateStore(state => state.selectedNodeId);

    const activeGenomeId = useCanvasGenomeStore(state => {
        if (selectedGenomeIdRaw) return selectedGenomeIdRaw;
        if (selectedNodeId) return state.nodes.get(selectedNodeId)?.genomeId || null;
        return null;
    });

    const selectedGenomeEntry = useCanvasGenomeStore(state => activeGenomeId ? state.genomes.get(activeGenomeId) : null);
    const selectedGenomeResources = selectedGenomeEntry ? selectedGenomeEntry.genome.GetGenomeResources() : null;

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

        const maxNodes = settings.useMaxNodesLimit ? settings.maxNodesLimit : undefined;

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
                mutatedResult = targetGenome.MutateAddNode(maxNodes);
            } catch (e) {
                console.error("Add Node Mutation failed:", e);
            }
        }

        const rSkip = Math.random();
        const rChange = Math.random();

        if (!mutatedResult && rSkip < settings.mutationRates.addSkipConnection) {
            console.log(`[Mutate Current] Rolled ${rSkip.toFixed(2)} < ${settings.mutationRates.addSkipConnection}. Firing AddSkipConnection mutation...`);
            try {
                mutatedResult = targetGenome.MutateAddSkipConnection(maxNodes);
            } catch (e) {
                console.error("Add Skip Connection Mutation failed:", e);
            }
        }

        if (!mutatedResult && rChange < settings.mutationRates.changeLayerType) {
            console.log(`[Mutate Current] Rolled ${rChange.toFixed(2)} < ${settings.mutationRates.changeLayerType}. Firing ChangeLayerType mutation...`);
            try {
                mutatedResult = targetGenome.MutateChangeLayerType(maxNodes);
            } catch (e) {
                console.error("Change Layer Type Mutation failed:", e);
            }
        }

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

        // Determine which active CrossoverStrategy to use
        const activeStrategies = settings.selectedCrossovers.filter(s =>
            s === 'subgraph-insertion' || s === 'subgraph-replacement' || s === 'neat-style' || s === 'multi-point'
        );

        if (activeStrategies.length === 0) {
            alert("Please enable at least one Crossover Strategy (Subgraph Insertion or Replacement).");
            return;
        }

        const chosenStrategy = activeStrategies[Math.floor(Math.random() * activeStrategies.length)];
        console.log(`[Breed Current] Selected strategy: ${chosenStrategy} between ${recipientIdx} and ${donorIdx}`);

        // Try breed 10 times to find valid insertion or replacement
        let result = null;
        for (let i = 0; i < 10; i++) {
            try {
                let breedResult = null;

                if (chosenStrategy === 'subgraph-replacement') {
                    breedResult = recipient.BreedByReplacement(donor, maxNodes);
                } else if (chosenStrategy === 'neat-style') {
                    breedResult = recipient.BreedNeatStyle(donor, maxNodes);
                } else if (chosenStrategy === 'multi-point') {
                    breedResult = recipient.BreedMultiPoint(donor, maxNodes);
                } else {
                    breedResult = recipient.Breed(donor, maxNodes);
                }

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

                <label className={styles.label} style={{ opacity: 0.5, cursor: 'not-allowed' }}>
                    <input type="checkbox" disabled checked={settings.useParsimonyPressure} onChange={e => settings.setUseParsimonyPressure(e.target.checked)} />
                    Parsimony Pressure (α)
                    {settings.useParsimonyPressure && (
                        <input type="number" step="0.001" className={styles.input} style={{ width: 60 }} value={settings.parsimonyAlpha} onChange={e => settings.setParsimonyAlpha(parseFloat(e.target.value) || 0)} disabled />
                    )}
                </label>

                <label className={styles.label} style={{ opacity: 0.5, cursor: 'not-allowed' }}>
                    <input type="checkbox" disabled checked={settings.useResourceAwareFitness} onChange={e => settings.setUseResourceAwareFitness(e.target.checked)} />
                    Resource-Aware Fitness
                </label>
                {settings.useResourceAwareFitness && (
                    <div style={{ marginLeft: 20, display: 'flex', flexDirection: 'column', gap: 4, opacity: 0.5, pointerEvents: 'none' }}>
                        <label className={styles.label} style={{ fontSize: '0.8em' }}>
                            Max Flash (bytes):
                            <input type="number" className={styles.input} style={{ width: 80, marginLeft: 8 }} value={settings.resourceTargets.flash} onChange={e => settings.setResourceTarget('flash', parseInt(e.target.value) || 0)} disabled />
                        </label>
                        <label className={styles.label} style={{ fontSize: '0.8em' }}>
                            Max RAM (bytes):
                            <input type="number" className={styles.input} style={{ width: 80, marginLeft: 8 }} value={settings.resourceTargets.ram} onChange={e => settings.setResourceTarget('ram', parseInt(e.target.value) || 0)} disabled />
                        </label>
                        <label className={styles.label} style={{ fontSize: '0.8em' }}>
                            Max MACs:
                            <input type="number" className={styles.input} style={{ width: 80, marginLeft: 8 }} value={settings.resourceTargets.macs} onChange={e => settings.setResourceTarget('macs', parseInt(e.target.value) || 0)} disabled />
                        </label>
                    </div>
                )}
            </div>

            <div className={styles.section}>
                <h4 className={styles.sectionTitle}>Actions</h4>
                <div className={styles.buttonGroup}>
                    <button className={styles.button} onClick={handleMutateCurrent}>Mutate Current</button>
                    <button className={styles.button} onClick={handleBreed}>Breed Random</button>
                </div>
                <div className={styles.metrics}>
                    <div>Current Nodes on Canvas: {canvasNodesCount}</div>
                    {selectedGenomeResources && (
                        <div style={{ marginTop: 10, fontSize: '0.85em', color: '#aaa' }}>
                            <strong>Selected Genome Metrics:</strong>
                            <div style={{ paddingLeft: 10 }}>Nodes: {selectedGenomeResources.totalNodes}</div>
                            <div style={{ paddingLeft: 10 }}>Flash: {selectedGenomeResources.totalFlash} bytes</div>
                            <div style={{ paddingLeft: 10 }}>RAM: {selectedGenomeResources.totalRam} bytes</div>
                            <div style={{ paddingLeft: 10 }}>MACs: {selectedGenomeResources.totalMacs.toLocaleString()}</div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
