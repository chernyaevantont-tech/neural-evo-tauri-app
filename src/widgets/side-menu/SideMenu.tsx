import { BsLayers } from "react-icons/bs";
import styles from './SideMenu.module.css'
import { useState } from "react";
import { NodeToolbar } from "../../features/node-toolbar";
import { PiGraph } from "react-icons/pi";
import { GenomeToolbar } from "../../features/genome-toolbar";
import { HiOutlineMenu } from "react-icons/hi";
import { RxCross1 } from "react-icons/rx";

type MenuType = "Layers" | "Genomes";

interface SideMenuProps {
    handleAddNode: (nodeType: string) => void;
    handleGetSubgenome: () => void;
    handleLoadGenome: () => void;
}

export const SideMenu: React.FC<SideMenuProps> = ({ handleAddNode, handleLoadGenome, handleGetSubgenome }) => {
    const [selectedPanelMenu, setSelectedPanelMenu] = useState<MenuType>("Layers");
    const [menuIsOpen, setMenuIsOpen] = useState<boolean>(false);

    return (
        <>
            <div className={styles.container}>
                <div
                    className={`${styles.iconContainer} ${styles.divider}`}
                    onClick={() => setMenuIsOpen(prev => !prev)}
                >
                    {
                        menuIsOpen
                            ?
                            <RxCross1 className={styles.icon} />
                            :
                            <HiOutlineMenu className={styles.icon} />
                    }
                </div>
                <div
                    className={`${styles.iconContainer} ${selectedPanelMenu == "Layers" && styles.selectedIcon}`}
                    onClick={() => setSelectedPanelMenu("Layers")}
                >
                    <BsLayers className={styles.icon} />
                </div>
                <div
                    className={`${styles.iconContainer} ${selectedPanelMenu == "Genomes" && styles.selectedIcon}`}
                    onClick={() => setSelectedPanelMenu("Genomes")}
                >
                    <PiGraph className={styles.icon} />
                </div>
                {
                    menuIsOpen && <div className={styles.sideMenu}>
                        {
                            selectedPanelMenu == "Layers" && <NodeToolbar onAddNode={handleAddNode} />
                        }
                        {
                            selectedPanelMenu == "Genomes" && <GenomeToolbar
                                onLoadGenome={handleLoadGenome}
                                onGetSubgenome={handleGetSubgenome}
                            />
                        }
                    </div>
                }
            </div>
        </>
    )
}