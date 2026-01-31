import React from "react";
import { VisualGenome } from "./types";

interface GenomeInfoPanelProps {
    genomes: VisualGenome[];
};

export const GenomeInfoPanel: React.FC<GenomeInfoPanelProps> = ({genomes}) => {
    if (genomes.length == 0) {
        return (
            <div style={panelStyle}>
                <h3 style={headerStyle}>Genomes Information</h3>
                <p style={emptyStyle}>Genomes not found</p>
            </div>
        );
    }

    return (
        <div style={panelStyle}>
            <h3 style={headerStyle}>Genomes Information</h3>
            {
                genomes.map(genome => (
                    <div style={{color: genome.isValid ? 'green' : 'red'}}>
                        {genome.id}
                    </div>
                ))
            }
        </div>
    )
}

const panelStyle: React.CSSProperties = {
    background: 'white',
    padding: '20px',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
    minWidth: '300px',
    maxWidth: '400px',
    maxHeight: '80vh',
    overflow: 'auto'
};

const headerStyle: React.CSSProperties = {
    margin: '0 0 20px 0',
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#333',
    borderBottom: '2px solid #2196F3',
    paddingBottom: '10px'
};

const emptyStyle: React.CSSProperties = {
    color: '#999',
    fontStyle: 'italic',
    textAlign: 'center',
    padding: '20px'
};
