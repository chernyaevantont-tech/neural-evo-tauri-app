import { BsLayers } from "react-icons/bs";
import styles from './SideMenu.module.css'
import { useState } from "react";
import { FcBiotech } from "react-icons/fc";
import { NodeToolbar } from "../../features/node-toolbar";

type MenuType = "Layers" | "Genomes";

export const SideMenu: React.FC = () => {
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
                    <FcBiotech className={styles.icon} />
                </div>
                {
                    selectedPanelMenu && <div className={styles.sideMenu}>
                        {
                            selectedPanelMenu == "Layers" && <NodeToolbar/>
                        }
                    </div>
                }
            </div>
        </>
    )
}