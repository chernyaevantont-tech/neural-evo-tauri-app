import React, { useState } from "react";
import styles from "./AddNodeToolbar.module.css";
import { Button, PlusIcon } from "../../../../shared";
import { AddNodeModal } from "../AddNodeModal/AddNodeModal";
import { BaseNode } from "../../../../entities/canvas-genome";
import { useAddNode } from "../../model/useAddNode";

export const AddNodeToolbar: React.FC = () => {
    const nodeTypes = [
        { type: 'Input', label: 'Input' },
        { type: 'Dense', label: 'Dense' },
        { type: 'Conv2D', label: 'Conv2D' },
        { type: 'Pooling', label: 'Pooling' },
        { type: 'Flatten', label: 'Flatten' },
        { type: 'Add', label: 'Add' },
        { type: 'Concat2D', label: 'Concat' },
        { type: 'Output', label: 'Output' },
    ];

    const [nodeType, setNodeType] = useState<string | null>(null);

    const addNode = useAddNode();

    const onAddNode = (type: string) => {
        setNodeType(type);
    }

    const onSave = (node: BaseNode) => {
        addNode(node);
        setNodeType(null);
    }

    const onCancel = () => {
        setNodeType(null);
    }

    return (
        <>
            <div className={styles.container}>
                <div className={styles.section}>
                    <h4 className={styles.sectionTitle}>Add Layers</h4>
                    <div className={styles.buttonGrid}>
                        {nodeTypes.map(({ type, label }) => (
                            <Button
                                key={type}
                                onClick={() => onAddNode(type)}
                                variant="secondary"
                                size="sm"
                                icon={<PlusIcon size={14} />}
                            >
                                {label}
                            </Button>
                        ))}
                    </div>
                </div>

                <div className={styles.divider} />
            </div>
            {
                nodeType && <AddNodeModal
                    nodeType={nodeType}
                    onSave={onSave}
                    onCancel={onCancel}
                />
            }
        </>
    )
}