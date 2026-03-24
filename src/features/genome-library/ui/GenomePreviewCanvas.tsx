import React, { useEffect, useState } from 'react';
import { useGenomeLibraryStore } from '../model/store';
import { BaseNode, deserializeGenome, GenomeSvgPreview } from '../../../entities/canvas-genome';

interface GenomePreviewCanvasProps {
    genomeId: string;
}

export const GenomePreviewCanvas: React.FC<GenomePreviewCanvasProps> = ({ genomeId }) => {
    const loadGenomeContent = useGenomeLibraryStore(s => s.loadGenomeContent);
    const [nodes, setNodes] = useState<BaseNode[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        let cancelled = false;

        const load = async () => {
            setLoading(true);
            try {
                const genomeStr = await loadGenomeContent(genomeId);
                const result = await deserializeGenome(genomeStr);

                if (cancelled) return;
                setNodes(result.nodes);
            } catch (err) {
                console.error('Preview load failed:', err);
            } finally {
                if (!cancelled) setLoading(false);
            }
        };

        load();
        return () => { cancelled = true; };
    }, [genomeId, loadGenomeContent]);

    if (loading) {
        return (
            <div style={{
                width: '100%', height: '100%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                opacity: 0.5,
            }}>
                Loading preview...
            </div>
        );
    }

    return <GenomeSvgPreview nodes={nodes} />;
};
