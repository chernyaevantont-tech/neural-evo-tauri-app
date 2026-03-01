import React, { useEffect, useState } from 'react';
import { Window } from '@tauri-apps/api/window';
import classes from './TitleBar.module.css'
import { BsDashLg, BsWindowFullscreen, BsWindowStack, BsXLg, BsHouse } from 'react-icons/bs';
import { useNavigate, useLocation } from 'react-router-dom';

export const TitleBar: React.FC = () => {
  const [appWindow, _] = useState(Window.getCurrent());
  const [maximizedFlag, setMaximizedFlag] = useState<boolean>(false);
  const navigate = useNavigate();
  const location = useLocation();

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
          {location.pathname !== '/' && (
            <div className={classes.HomeButton} onClick={() => navigate('/')}>
              <BsHouse className={classes.IconLabel} />
              <span>Hub</span>
            </div>
          )}
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