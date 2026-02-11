import { BsLayers } from "react-icons/bs";
import styles from './SideMenu.module.css'
import { Dispatch, SetStateAction, useState } from "react";
import { NodeToolbar } from "../../features/node-toolbar";
import { PiGraph } from "react-icons/pi";
import { GenomeToolbar } from "../../features/genome-toolbar";
import { HiOutlineMenu } from "react-icons/hi";
import { RxCross1 } from "react-icons/rx";

export type MenuType = "Layers" | "Genomes";

interface SideMenuProps {
    handleAddNode: (nodeType: string) => void;
    handleGetSubgenome: () => void;
    handleLoadGenome: () => void;
    menuType: MenuType;
    setMenuType: Dispatch<SetStateAction<MenuType>>;
}

export const SideMenu: React.FC<SideMenuProps> = ({ 
    handleAddNode, 
    handleLoadGenome, 
    handleGetSubgenome,
    menuType,
    setMenuType
}) => {
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
                    className={`${styles.iconContainer} ${menuType == "Layers" && styles.selectedIcon}`}
                    onClick={() => setMenuType("Layers")}
                >
                    <BsLayers className={styles.icon} />
                </div>
                <div
                    className={`${styles.iconContainer} ${menuType == "Genomes" && styles.selectedIcon}`}
                    onClick={() => setMenuType("Genomes")}
                >
                    <PiGraph className={styles.icon} />
                </div>
                {
                    menuIsOpen && <div className={styles.sideMenu}>
                        {
                            menuType == "Layers" && <NodeToolbar onAddNode={handleAddNode} />
                        }
                        {
                            menuType == "Genomes" && <GenomeToolbar
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