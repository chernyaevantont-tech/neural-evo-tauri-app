import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { HiddenArchivePage, filterAndSortHiddenEntries } from './HiddenArchivePage';
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

const makeEntry = (id: string, overrides?: Partial<GenomeLibraryEntry>): GenomeLibraryEntry => ({
    id,
    name: `Hidden ${id}`,
    tags: ['hidden'],
    createdAt: '2026-03-24T12:00:00.000Z',
    inputDims: [3],
    outputDims: [1],
    totalNodes: 11,
    layerTypes: ['Dense'],
    bestLoss: 0.3,
    bestAccuracy: 0.8,
    sourceGeneration: 3,
    parentGenomes: ['p-1', 'p-2'],
    createdAtUnixMs: 200,
    fitnessMetrics: {
        loss: 0.3,
        accuracy: 0.8,
        inferenceLatencyMs: 12,
        modelSizeMb: 1.8,
    },
    ...overrides,
});

describe('HiddenArchivePage', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        listHiddenLibraryMock.mockResolvedValue([]);
        unhideHiddenGenomeMock.mockResolvedValue(undefined);
        deleteHiddenGenomeMock.mockResolvedValue(undefined);
        exportGenomeWithWeightsMock.mockResolvedValue({});
        pickFolderMock.mockResolvedValue('C:/tmp/export');
        getGenealogyPathMock.mockResolvedValue({ records: [{ genome_id: 'a' }, { genome_id: 'b' }] });

        vi.spyOn(window, 'confirm').mockReturnValue(true);
    });

    it('filter and sort helper works for search and latency sorting', () => {
        const entries = [
            makeEntry('g-1', { fitnessMetrics: { loss: 0.1, accuracy: 0.95, inferenceLatencyMs: 5, modelSizeMb: 1 } }),
            makeEntry('g-2', { fitnessMetrics: { loss: 0.2, accuracy: 0.85, inferenceLatencyMs: 15, modelSizeMb: 2 } }),
        ];

        const result = filterAndSortHiddenEntries(entries, 'g-2', 'latency_asc');

        expect(result).toHaveLength(1);
        expect(result[0].id).toBe('g-2');
    });

    it('handles selection and batch unhide action', async () => {
        listHiddenLibraryMock.mockResolvedValueOnce([makeEntry('g-1'), makeEntry('g-2')]);

        render(<HiddenArchivePage />);

        await screen.findByText('g-1');
        fireEvent.click(screen.getByLabelText('Select g-1'));
        fireEvent.click(screen.getByRole('button', { name: 'Unhide selected' }));

        await waitFor(() => {
            expect(unhideHiddenGenomeMock).toHaveBeenCalledWith('g-1');
        });

        expect(screen.getByText(/Unhide selected: success 1/i)).toBeTruthy();
    });

    it('opens detail modal and loads genealogy path', async () => {
        listHiddenLibraryMock.mockResolvedValueOnce([makeEntry('g-42')]);

        render(<HiddenArchivePage />);

        await screen.findByText('g-42');
        fireEvent.click(screen.getByRole('button', { name: /Details/i }));

        expect(screen.getByText(/Hidden Genome: g-42/)).toBeTruthy();

        fireEvent.click(screen.getByRole('button', { name: 'Open genealogy' }));

        await waitFor(() => {
            expect(getGenealogyPathMock).toHaveBeenCalledWith('g-42');
        });
    });

    it('shows empty state', async () => {
        listHiddenLibraryMock.mockResolvedValueOnce([]);

        render(<HiddenArchivePage />);

        await screen.findByText('No hidden genomes found');
        expect(screen.getByText('No hidden genomes found')).toBeTruthy();
    });
});
