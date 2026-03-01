import React from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './HomePage.module.css';
import { BsLayers, BsDatabase, BsLightningCharge, BsCollection } from 'react-icons/bs';

import { TitleBar } from '../../widgets/title-bar/TitleBar';

export const HomePage: React.FC = () => {
    const navigate = useNavigate();

    return (
        <>
            <TitleBar />
            <div className={styles.container}>
                <div className={styles.content}>
                    <div className={styles.header}>
                        <h1 className={styles.title}>Neural Evo Studio</h1>
                        <p className={styles.subtitle}>Welcome back. Choose a workspace to begin.</p>
                    </div>

                    <div className={styles.grid}>
                        {/* Sandbox Tile */}
                        <div
                            className={styles.card}
                            onClick={() => navigate('/sandbox')}
                        >
                            <div className={styles.cardHeader}>
                                <div className={styles.iconWrapper}>
                                    <BsLayers />
                                </div>
                                <h2 className={styles.cardTitle}>Architecture Sandbox</h2>
                            </div>
                            <p className={styles.cardDescription}>
                                Manually design, mutate, and evaluate neural network graphs in a freeform node-based editor.
                            </p>
                            <div className={styles.cardFooter}>
                                <span>Launch Editor &rarr;</span>
                            </div>
                        </div>

                        {/* Dataset Manager Tile */}
                        <div
                            className={styles.card}
                            onClick={() => navigate('/dataset-manager')}
                        >
                            <div className={styles.cardHeader}>
                                <div className={styles.iconWrapper} style={{ color: 'var(--color-warning)' }}>
                                    <BsDatabase />
                                </div>
                                <h2 className={styles.cardTitle}>Dataset Manager</h2>
                            </div>
                            <p className={styles.cardDescription}>
                                Import and configure training datasets from generic formats (MNIST, CIFA-10, Folder image bins, CSV).
                            </p>
                            <div className={styles.cardFooter}>
                                <span style={{ color: 'var(--color-warning)' }}>Manage Datasets &rarr;</span>
                            </div>
                        </div>

                        {/* Evolution Studio Tile */}
                        <div
                            className={styles.card}
                            onClick={() => navigate('/evolution-studio')}
                        >
                            <div className={styles.cardHeader}>
                                <div className={styles.iconWrapper} style={{ color: 'var(--color-success)' }}>
                                    <BsLightningCharge />
                                </div>
                                <h2 className={styles.cardTitle}>Evolution Studio</h2>
                            </div>
                            <p className={styles.cardDescription}>
                                Configure and run full Neural Architecture Search pipelines with distributed Rust hardware evaluation.
                            </p>
                            <div className={styles.cardFooter}>
                                <span style={{ color: 'var(--color-success)' }}>Launch Studio &rarr;</span>
                            </div>
                        </div>

                        {/* Genome Library Tile */}
                        <div
                            className={styles.card}
                            onClick={() => navigate('/genome-library')}
                        >
                            <div className={styles.cardHeader}>
                                <div className={styles.iconWrapper} style={{ color: '#9b59b6' }}>
                                    <BsCollection />
                                </div>
                                <h2 className={styles.cardTitle}>Genome Library</h2>
                            </div>
                            <p className={styles.cardDescription}>
                                Browse, search, and manage saved neural network architectures. Preview graphs and check compatibility.
                            </p>
                            <div className={styles.cardFooter}>
                                <span style={{ color: '#9b59b6' }}>Browse Genomes &rarr;</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};
