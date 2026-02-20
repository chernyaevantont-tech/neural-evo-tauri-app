import styles from './ContextMenu.module.css';

interface ContextMenuProps {
    x: number;
    y: number;
    children: React.ReactNode;
}

const ContextMenuRoot: React.FC<ContextMenuProps> = ({
    x,
    y,
    children,
}) => {
    return (<div
        className={styles.container}
        style={{ left: x, top: y }}
        onMouseDown={(e) => e.stopPropagation()}
    >
        {children}
    </div>)
}

interface MenuItemProps {
    icon: React.ReactNode;
    label: string;
    onClick: () => void;
    cancelContextMenu?: () => void;
    danger?: boolean;
}

const MenuItem: React.FC<MenuItemProps> = ({
    icon,
    label,
    onClick,
    cancelContextMenu,
    danger
}) => {
    return (
        <button
            onClick={() => {onClick(); cancelContextMenu && cancelContextMenu();}}
            className={`${styles.menuItem} ${danger ? styles.danger : ''}`}
        >
            <span className={styles.icon}>{icon}</span>
            {label}
        </button>
    )
}

export const ContextMenu = Object.assign(ContextMenuRoot, {
    MenuItem: MenuItem,
});