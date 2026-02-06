import React, { useEffect, useState } from 'react';
import { Window } from '@tauri-apps/api/window';
import classes from './TitleBar.module.css'
import { BsDashLg, BsWindowFullscreen, BsWindowStack, BsXLg } from 'react-icons/bs';

export const TitleBar: React.FC = () => {
  const [appWindow, _] = useState(Window.getCurrent());
  const [maximizedFlag, setMaximizedFlag] = useState<boolean>(false);

  useEffect(() => {
    const handleResize = () => {
      appWindow.isMaximized().then(resp => setMaximizedFlag(resp));
    }
    handleResize();

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return (
    <>
      <header className={`${classes.AppTitle} ${maximizedFlag ? classes.AppTitleMaximized : ""}`}>
        <div className={classes.Left} data-tauri-drag-region>

        </div>
        <div className={classes.Right} data-tauri-drag-region>
          <BsDashLg onClick={() => appWindow.minimize()} className={classes.Icon} />
          {
            maximizedFlag ?
              <BsWindowStack className={classes.Icon} onClick={() => appWindow.unmaximize()} />
              :
              <BsWindowFullscreen className={classes.Icon} onClick={() => appWindow.maximize()} />
          }
          <BsXLg className={`${classes.Icon} ${classes.Quit}`} onClick={() => appWindow.close()} />
        </div>
      </header>
      {/* <Outlet /> */}
    </>
  )
};