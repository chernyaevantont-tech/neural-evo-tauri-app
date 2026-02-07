import { BsLayers } from "react-icons/bs";
import styles from './SideMenu.module.css'
import { useState } from "react";
import { NodeToolbar } from "../../features/node-toolbar";
import { PiGraph } from "react-icons/pi";

type MenuType = "Layers" | "Genomes";

interface SideMenuProps {
    handleAddNode: (nodeType: string) => void;
}

export const SideMenu: React.FC<SideMenuProps> = ({handleAddNode}) => {
    const [selectedPanelMenu, setSelectedPanelMenu] = useState<MenuType | null>(null);

    return (
        <>
            <div className={styles.container}>
                <div
                    className={`${styles.iconContainer} ${selectedPanelMenu == "Layers" && styles.selectedIcon}`}
                    onClick={() => setSelectedPanelMenu(selectedPanelMenu == "Layers" ? null : "Layers")}
                >
                    <BsLayers className={styles.icon} />
                </div>
                <div
                    className={`${styles.iconContainer} ${selectedPanelMenu == "Genomes" && styles.selectedIcon}`}
                    onClick={() => setSelectedPanelMenu(selectedPanelMenu == "Genomes" ? null : "Genomes")}
                >
                    <PiGraph  className={styles.icon} />
                </div>
                {
                    selectedPanelMenu && <div className={styles.sideMenu}>
                        {
                            selectedPanelMenu == "Layers" && <NodeToolbar onAddNode={handleAddNode}/>
                        }
                    </div>
                }
            </div>
        </>
    )
}