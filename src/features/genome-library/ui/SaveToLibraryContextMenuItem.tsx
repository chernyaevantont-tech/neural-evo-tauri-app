import React, { useState } from 'react';
import { BsBoxArrowInDown } from 'react-icons/bs';
import { useCanvasStateStore } from '../../../entities/canvas-state';
import { useCanvasGenomeStore } from '../../../entities/canvas-genome/model/store';
import { serializeGenome } from '../../../entities/canvas-genome/lib/serializeGenome';
import { ContextMenu } from '../../../shared/ui/ContextMenu/ContextMenu';
import { useGenomeLibraryStore } from '../model/store';

interface SaveToLibraryContextMenuItemProps {
    cancelContextMenu: () => void;
}

export const SaveToLibraryContextMenuItem: React.FC<SaveToLibraryContextMenuItemProps> = ({ cancelContextMenu }) => {
    const contextMenuGenomeId = useCanvasStateStore(state => state.genomeContextMenu?.genomeId);
    const genomes = useCanvasGenomeStore(s => s.genomes);
    const saveGenome = useGenomeLibraryStore(s => s.saveGenome);

    const onClick = async () => {
        if (!contextMenuGenomeId) return;
        const visualGenome = genomes.get(contextMenuGenomeId);
        if (!visualGenome) return;

        const name = prompt('Genome name:', 'My Genome') || 'Unnamed Genome';
        const tagsStr = prompt('Tags (comma separated):', '') || '';
        const tags = tagsStr.split(',').map(t => t.trim()).filter(Boolean);

        try {
            const genomeStr = await serializeGenome(visualGenome.genome);
            await saveGenome(genomeStr, name, tags);
            alert('Genome saved to library!');
        } catch (err) {
            console.error('Failed to save to library:', err);
            alert('Failed to save: ' + String(err));
        }
    };

    return (
        <ContextMenu.MenuItem
            icon={<BsBoxArrowInDown size={14} />}
            label="Save to Library"
            cancelContextMenu={cancelContextMenu}
            onClick={onClick}
        />
    );
};
