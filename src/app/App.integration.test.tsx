import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';

vi.mock('../pages/home-page/HomePage', () => ({
    HomePage: () => <div>HomePageMock</div>,
}));

vi.mock('../pages/network-editor-page/NetworkEditorPage', () => ({
    NetworkEditorPage: () => <div>NetworkEditorPageMock</div>,
}));

vi.mock('../pages/dataset-manager-page/DatasetManagerPage', () => ({
    DatasetManagerPage: () => <div>DatasetManagerPageMock</div>,
}));

vi.mock('../pages/evolution-studio-page/EvolutionStudioPage', () => ({
    EvolutionStudioPage: () => <div>EvolutionStudioPageMock</div>,
}));

vi.mock('../pages/genome-library-page/GenomeLibraryPage', () => ({
    GenomeLibraryPage: () => <div>GenomeLibraryPageMock</div>,
}));

vi.mock('../pages/hidden-archive-page', () => ({
    HiddenArchivePage: () => <div>HiddenArchivePageMock</div>,
}));

function renderAt(path: string) {
    window.history.pushState({}, '', path);
    return render(<App />);
}

describe('App route integration', () => {
    afterEach(() => {
        window.history.pushState({}, '', '/');
    });

    it('renders evolution studio route', () => {
        renderAt('/evolution-studio');
        expect(screen.getByText('EvolutionStudioPageMock')).toBeTruthy();
    });

    it('renders genome library route', () => {
        renderAt('/genome-library');
        expect(screen.getByText('GenomeLibraryPageMock')).toBeTruthy();
    });

    it('renders hidden archive route', () => {
        renderAt('/hidden-archive');
        expect(screen.getByText('HiddenArchivePageMock')).toBeTruthy();
    });
});
