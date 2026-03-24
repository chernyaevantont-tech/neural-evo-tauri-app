import React from 'react';
import styles from './DatasetManagerPage.module.css';
import type { DatasetValidationReport } from '../../features/dataset-manager';
import { BsCheckCircle, BsExclamationTriangle, BsExclamationCircle } from 'react-icons/bs';

interface Props {
    validationReport: DatasetValidationReport;
}

export const DatasetValidationPanel: React.FC<Props> = ({ validationReport }) => {
    // Be defensive: persisted profiles may contain partial/legacy validation payloads.
    const issues = validationReport?.issues ?? [];
    const inputShapes = validationReport?.input_shapes ?? {};
    const outputShape = validationReport?.output_shape;
    const totalValidSamples = validationReport?.total_valid_samples ?? 0;
    const canStartEvolution = validationReport?.can_start_evolution ?? false;
    const isValid = validationReport?.is_valid ?? false;

    const hasErrors = issues.some(i => i.severity === 'ERROR');
    const hasWarnings = issues.some(i => i.severity === 'WARNING');

    return (
        <div style={{
            background: 'var(--color-bg-primary)',
            border: '1px solid var(--color-border)',
            borderRadius: '8px',
            padding: '1.5rem',
            marginTop: '1rem'
        }}>
            {/* Header with status */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
                {isValid ? (
                    <>
                        <BsCheckCircle color="var(--color-success)" size={24} />
                        <div>
                            <h3 style={{ margin: '0 0 0.25rem 0', color: 'var(--color-success)' }}>
                                ✓ Dataset Valid
                            </h3>
                            <p style={{ margin: 0, color: 'var(--color-text-muted)', fontSize: '0.9rem' }}>
                                All streams configured correctly. Ready for evolution.
                            </p>
                        </div>
                    </>
                ) : (
                    <>
                        {hasErrors ? (
                            <>
                                <BsExclamationCircle color="var(--color-danger)" size={24} />
                                <div>
                                    <h3 style={{ margin: '0 0 0.25rem 0', color: 'var(--color-danger)' }}>
                                        ✕ Configuration Error
                                    </h3>
                                    <p style={{ margin: 0, color: 'var(--color-text-muted)', fontSize: '0.9rem' }}>
                                        Fix errors below before starting evolution.
                                    </p>
                                </div>
                            </>
                        ) : (
                            <>
                                <BsExclamationTriangle color="var(--color-warning)" size={24} />
                                <div>
                                    <h3 style={{ margin: '0 0 0.25rem 0', color: 'var(--color-warning)' }}>
                                        ⚠ Warnings
                                    </h3>
                                    <p style={{ margin: 0, color: 'var(--color-text-muted)', fontSize: '0.9rem' }}>
                                        Review warnings below before proceeding.
                                    </p>
                                </div>
                            </>
                        )}
                    </>
                )}
            </div>

            {/* Shapes Summary */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '1rem',
                marginBottom: '1.5rem',
                padding: '1rem',
                background: 'var(--color-bg-secondary)',
                borderRadius: '6px'
            }}>
                {/* Input Shapes */}
                <div>
                    <h4 style={{ margin: '0 0 0.75rem 0', color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
                        Input Shape(s)
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {Object.entries(inputShapes).length > 0 ? (
                            Object.entries(inputShapes).map(([streamId, shape]) => (
                                <div key={streamId} style={{
                                    background: 'var(--color-bg-tertiary)',
                                    padding: '0.5rem 0.75rem',
                                    borderRadius: '4px',
                                    fontFamily: 'monospace',
                                    fontSize: '0.85rem',
                                    color: 'var(--color-text-primary)'
                                }}>
                                    [{shape.join(', ')}]
                                </div>
                            ))
                        ) : (
                            <p style={{ margin: 0, color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>
                                Not detected
                            </p>
                        )}
                    </div>
                </div>

                {/* Output Shape */}
                <div>
                    <h4 style={{ margin: '0 0 0.75rem 0', color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
                        Output Shape
                    </h4>
                    {outputShape ? (
                        <div style={{
                            background: 'var(--color-bg-tertiary)',
                            padding: '0.5rem 0.75rem',
                            borderRadius: '4px',
                            fontFamily: 'monospace',
                            fontSize: '0.85rem',
                            color: 'var(--color-text-primary)'
                        }}>
                            [{outputShape.join(', ')}]
                        </div>
                    ) : (
                        <p style={{ margin: 0, color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>
                            Not detected
                        </p>
                    )}

                    <p style={{
                        margin: '0.5rem 0 0 0',
                        color: 'var(--color-text-muted)',
                        fontSize: '0.75rem'
                    }}>
                        {totalValidSamples} valid samples
                    </p>
                </div>
            </div>

            {/* Issues List */}
            {issues.length > 0 && (
                <div>
                    <h4 style={{ margin: '0 0 1rem 0', color: 'var(--color-text-secondary)' }}>
                        Issues ({issues.length})
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        {issues.map((issue, idx) => (
                            <div key={idx} style={{
                                background: issue.severity === 'ERROR'
                                    ? 'rgba(231, 76, 60, 0.1)'
                                    : 'rgba(241, 196, 15, 0.1)',
                                border: `1px solid ${issue.severity === 'ERROR'
                                    ? 'var(--color-danger)'
                                    : 'var(--color-warning)'}`,
                                borderRadius: '6px',
                                padding: '1rem'
                            }}>
                                {/* Issue Header */}
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'flex-start',
                                    gap: '0.75rem',
                                    marginBottom: issue.suggested_fix ? '0.75rem' : 0
                                }}>
                                    {issue.severity === 'ERROR' ? (
                                        <BsExclamationCircle
                                            color="var(--color-danger)"
                                            size={18}
                                            style={{ marginTop: '2px', flexShrink: 0 }}
                                        />
                                    ) : (
                                        <BsExclamationTriangle
                                            color="var(--color-warning)"
                                            size={18}
                                            style={{ marginTop: '2px', flexShrink: 0 }}
                                        />
                                    )}
                                    <div style={{ flex: 1 }}>
                                        <p style={{
                                            margin: '0 0 0.25rem 0',
                                            color: issue.severity === 'ERROR'
                                                ? 'var(--color-danger)'
                                                : 'var(--color-warning)',
                                            fontWeight: 600
                                        }}>
                                            {issue.component}
                                        </p>
                                        <p style={{
                                            margin: 0,
                                            color: 'var(--color-text-primary)',
                                            fontSize: '0.9rem',
                                            lineHeight: 1.4
                                        }}>
                                            {issue.message}
                                        </p>
                                    </div>
                                </div>

                                {/* Suggested Fix */}
                                {issue.suggested_fix && (
                                    <div style={{
                                        background: 'var(--color-bg-primary)',
                                        borderLeft: '3px solid var(--color-accent-primary)',
                                        padding: '0.75rem',
                                        borderRadius: '4px',
                                        marginLeft: '26px'
                                    }}>
                                        <p style={{
                                            margin: '0 0 0.5rem 0',
                                            color: 'var(--color-text-secondary)',
                                            fontSize: '0.85rem',
                                            fontWeight: 600
                                        }}>
                                            Suggested fix:
                                        </p>
                                        <p style={{
                                            margin: 0,
                                            color: 'var(--color-text-primary)',
                                            fontSize: '0.85rem',
                                            lineHeight: 1.4
                                        }}>
                                            {issue.suggested_fix}
                                        </p>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* No issues message */}
            {issues.length === 0 && (
                <div style={{
                    padding: '1rem',
                    background: 'var(--color-bg-secondary)',
                    borderRadius: '6px',
                    textAlign: 'center'
                }}>
                    <p style={{
                        margin: 0,
                        color: 'var(--color-text-muted)',
                        fontSize: '0.9rem'
                    }}>
                        No issues found. Your dataset is properly configured. ✓
                    </p>
                </div>
            )}

            {/* Footer */}
            <div style={{
                marginTop: '1.5rem',
                paddingTop: '1.5rem',
                borderTop: '1px solid var(--color-border)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <p style={{
                    margin: 0,
                    color: 'var(--color-text-muted)',
                    fontSize: '0.85rem'
                }}>
                    {canStartEvolution ? (
                        <span style={{ color: 'var(--color-success)' }}>✓ Ready for evolution</span>
                    ) : (
                        <span style={{ color: 'var(--color-danger)' }}>✕ Cannot start evolution yet</span>
                    )}
                </p>
            </div>
        </div>
    );
};
