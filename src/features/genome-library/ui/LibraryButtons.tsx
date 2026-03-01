import React, { useState } from 'react';
import { BsCollection } from 'react-icons/bs';
import { Button } from '../../../shared';
import { useCanvasGenomeStore } from '../../../entities/canvas-genome/model/store';
import { useCanvasStateStore } from '../../../entities/canvas-state/model/store';
import { deserializeGenome } from '../../../entities/canvas-genome/lib/deserializeGenome';
import { useGenomeLibraryStore, GenomeLibraryEntry } from '../model/store';
import { GenomeCatalogPicker } from './GenomeCatalogPicker';

/**
 * Load a genome from the Library onto the canvas.
 * Uses the reusable GenomeCatalogPicker modal.
 */
export const LoadFromLibraryButton: React.FC = () => {
    const [showPicker, setShowPicker] = useState(false);
    const { loadGenomeContent } = useGenomeLibraryStore();
    const addNewGenome = useCanvasGenomeStore(s => s.addGenome);
    const translate = useCanvasStateStore(s => s.translate);
    const scale = useCanvasStateStore(s => s.scale);
    const canvasWidth = useCanvasStateStore(s => s.canvasWidth);
    const canvasHeight = useCanvasStateStore(s => s.canvasHeight);

    const handleSelect = async (entry: GenomeLibraryEntry) => {
        try {
            const genomeStr = await loadGenomeContent(entry.id);
            const { nodes, genome } = await deserializeGenome(genomeStr);
            addNewGenome(nodes, genome, canvasWidth, canvasHeight, translate.x, translate.y, scale, 300);
            setShowPicker(false);
        } catch (err) {
            console.error('Failed to load from library:', err);
        }
    };

    return (
        <>
            <Button
                onClick={() => setShowPicker(true)}
                variant="secondary"
                size="md"
                icon={<BsCollection size={16} />}
                fullWidth
            >
                Load from Library
            </Button>

            {showPicker && (
                <GenomeCatalogPicker
                    title="Load from Library"
                    onSelect={handleSelect}
                    onClose={() => setShowPicker(false)}
                />
            )}
        </>
    );
};

