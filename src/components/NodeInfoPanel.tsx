import React from 'react';
import { VisualNode } from './types';

interface NodeInfoPanelProps {
    selectedNode: VisualNode | null;
}

export const NodeInfoPanel: React.FC<NodeInfoPanelProps> = ({ selectedNode }) => {
    if (!selectedNode) {
        return (
            <div style={panelStyle}>
                <h3 style={headerStyle}>Node Information</h3>
                <p style={emptyStyle}>Select a node to view its information</p>
            </div>
        );
    }

    let info: any;
    try {
        info = JSON.parse(selectedNode.node.GetInfo() as string);
    } catch (e) {
        info = { error: 'Failed to parse node info' };
    }

    const renderValue = (value: any): string => {
        if (Array.isArray(value)) {
            return `[${value.join(', ')}]`;
        }
        if (typeof value === 'object' && value !== null) {
            return JSON.stringify(value, null, 2);
        }
        return String(value);
    };

    return (
        <div style={panelStyle}>
            <h3 style={headerStyle}>Node Information</h3>
            
            <div style={sectionStyle}>
                <div style={labelStyle}>Type:</div>
                <div style={valueStyle}>{info.node}</div>
            </div>

            {info.params && (
                <div style={sectionStyle}>
                    <div style={labelStyle}>Parameters:</div>
                    <div style={paramsContainerStyle}>
                        {Object.entries(info.params).map(([key, value]) => (
                            <div key={key} style={paramRowStyle}>
                                <span style={paramKeyStyle}>{key}:</span>
                                <span style={paramValueStyle}>{renderValue(value)}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div style={sectionStyle}>
                <div style={labelStyle}>Node ID:</div>
                <div style={{ ...valueStyle, fontSize: '10px', color: '#666' }}>
                    {selectedNode.node.id}
                </div>
            </div>

            <div style={sectionStyle}>
                <div style={labelStyle}>Output Shape:</div>
                <div style={valueStyle}>
                    {renderValue(selectedNode.node.GetOutputShape())}
                </div>
            </div>
        </div>
    );
};

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

const sectionStyle: React.CSSProperties = {
    marginBottom: '15px'
};

const labelStyle: React.CSSProperties = {
    fontSize: '12px',
    fontWeight: 'bold',
    color: '#666',
    textTransform: 'uppercase',
    marginBottom: '5px'
};

const valueStyle: React.CSSProperties = {
    fontSize: '14px',
    color: '#333',
    padding: '8px',
    background: '#f5f5f5',
    borderRadius: '4px',
    wordBreak: 'break-all'
};

const paramsContainerStyle: React.CSSProperties = {
    background: '#f5f5f5',
    borderRadius: '4px',
    padding: '10px'
};

const paramRowStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '5px 0',
    borderBottom: '1px solid #e0e0e0'
};

const paramKeyStyle: React.CSSProperties = {
    fontWeight: '500',
    color: '#555',
    fontSize: '13px'
};

const paramValueStyle: React.CSSProperties = {
    color: '#333',
    fontSize: '13px',
    textAlign: 'right',
    maxWidth: '60%',
    wordBreak: 'break-word'
};
