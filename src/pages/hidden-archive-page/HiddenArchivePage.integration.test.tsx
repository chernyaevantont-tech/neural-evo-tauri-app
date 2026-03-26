import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { HiddenArchivePage } from './HiddenArchivePage';
import type { GenomeLibraryEntry } from '../../features/genome-library';

const navigateMock = vi.fn();
const listHiddenLibraryMock = vi.fn();
const unhideHiddenGenomeMock = vi.fn();
const deleteHiddenGenomeMock = vi.fn();
const exportGenomeWithWeightsMock = vi.fn();
const pickFolderMock = vi.fn();
const getGenealogyPathMock = vi.fn();

vi.mock('react-router-dom', () => ({
    useNavigate: () => navigateMock,
}));

vi.mock('../../widgets/title-bar/TitleBar', () => ({
    TitleBar: () => <div>TitleBar</div>,
}));

vi.mock('../../features/evolution-studio/ui/ExportGenomeWithWeightsModal', () => ({
    ExportGenomeWithWeightsModal: () => <div>ExportModal</div>,
}));

vi.mock('../../features/genome-library', () => ({
    useGenomeLibraryStore: () => ({
        listHiddenLibrary: listHiddenLibraryMock,
        unhideHiddenGenome: unhideHiddenGenomeMock,
        deleteHiddenGenome: deleteHiddenGenomeMock,
        exportGenomeWithWeights: exportGenomeWithWeightsMock,
        pickFolder: pickFolderMock,
        getGenealogyPath: getGenealogyPathMock,
    }),
}));

function autosaveEntry(id: string): GenomeLibraryEntry {
    return {
        id,
        name: `Hidden ${id}`,
        tags: ['hidden', 'autosave'],
        createdAt: '2026-03-26T12:00:00.000Z',
        inputDims: [3],
        outputDims: [1],
        totalNodes: 8,
        layerTypes: ['Dense'],
        bestLoss: 0.2,
        bestAccuracy: 0.88,
        sourceGeneration: 4,
        parentGenomes: ['g-parent-a', 'g-parent-b'],
        createdAtUnixMs: 400,
        fitnessMetrics: {
            loss: 0.2,
            accuracy: 0.88,
            inferenceLatencyMs: 9,
            modelSizeMb: 1.2,
            trainingTimeMs: 150,
        },
    };
}

describe('HiddenArchivePage integration', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        listHiddenLibraryMock.mockResolvedValue([autosaveEntry('g-auto-1')]);
        unhideHiddenGenomeMock.mockResolvedValue(undefined);
        deleteHiddenGenomeMock.mockResolvedValue(undefined);
        exportGenomeWithWeightsMock.mockResolvedValue({});
        pickFolderMock.mockResolvedValue('C:/tmp/export');
        getGenealogyPathMock.mockResolvedValue({ records: [{ genome_id: 'a' }, { genome_id: 'b' }] });
        vi.spyOn(window, 'confirm').mockReturnValue(true);
    });

    it('covers hidden autosave to archive workflow with actionable unhide', async () => {
        render(<HiddenArchivePage />);

        await screen.findByText('g-auto-1');

        fireEvent.click(screen.getByLabelText('Select g-auto-1'));
        fireEvent.click(screen.getByRole('button', { name: 'Unhide selected' }));

        await waitFor(() => {
            expect(unhideHiddenGenomeMock).toHaveBeenCalledWith('g-auto-1');
        });
    });
});
