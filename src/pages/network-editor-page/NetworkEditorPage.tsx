import { useState } from 'react';
import styles from './NetworkEditorPage.module.css';
import { NetworkCanvas, SidePanel, TitleBar } from '../../widgets';
import { MenuType, SideMenu } from '../../widgets/side-menu/SideMenu';

export const NetworkEditorPage: React.FC = () => {
    const [menuType, setMenuType] = useState<MenuType>("Layers");


    return (
        <div className={styles.container}>
            <TitleBar />

            <div className={styles.content}>
                <SideMenu

                    menuType={menuType}
                    setMenuType={setMenuType}
                />

                <NetworkCanvas
                    menuType={menuType}
                />

                <SidePanel
                    menuType={menuType}
                />
            </div>
        </div>
    );
}