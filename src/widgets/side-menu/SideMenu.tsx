import { BsLayers } from "react-icons/bs";
import styles from './SideMenu.module.css'
import { Dispatch, SetStateAction, useState } from "react";
import { PiGraph } from "react-icons/pi";
import { HiOutlineMenu } from "react-icons/hi";
import { RxCross1 } from "react-icons/rx";
import { AddNodeToolbar } from "../../features/add-node";
import { LoadGenomeButton } from "../../features/genome-save-load/ui/LoadGenomeButton";
import { EvolutionManager } from "../../features/evolution-manager";

export type MenuType = "Layers" | "Genomes";

interface SideMenuProps {
    menuType: MenuType;
    setMenuType: Dispatch<SetStateAction<MenuType>>;
}

export const SideMenu: React.FC<SideMenuProps> = ({
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
                            menuType == "Layers" && <AddNodeToolbar />
                        }
                        {
                            menuType == "Genomes" && <div className={styles.genomeToolbarContainer}>
                                <div className={styles.section}>
                                    <h4 className={styles.sectionTitle}>Genome Operations</h4>
                                    <div className={styles.operations}>
                                        <LoadGenomeButton />
                                    </div>
                                </div>
                                <EvolutionManager />
                            </div>
                        }
                    </div>
                }
            </div>
        </>
    )
}