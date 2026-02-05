import React from 'react';
import { getCurrentWindow } from '@tauri-apps/api/window';

export const TitleBar: React.FC = () => {
  let appWindow: any = null;
  
  try {
    appWindow = getCurrentWindow();
  } catch (error) {
    console.error('Failed to get current window:', error);
  }

  const handleMinimize = async () => {
    if (!appWindow) return;
    try {
      await appWindow.minimize();
    } catch (error) {
      console.error('Failed to minimize:', error);
    }
  };

  const handleMaximize = async () => {
    if (!appWindow) return;
    try {
      await appWindow.toggleMaximize();
    } catch (error) {
      console.error('Failed to maximize:', error);
    }
  };

  const handleClose = async () => {
    if (!appWindow) return;
    try {
      await appWindow.close();
    } catch (error) {
      console.error('Failed to close:', error);
    }
  };

  console.log('TitleBar rendering');

  return (
    <div style={titleBarStyle} data-tauri-drag-region>
      <div style={titleStyle}>Neural Evolution Designer</div>
      <div style={buttonsContainerStyle}>
        <button
          style={buttonStyle}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#2a2d2e')}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
          onClick={handleMinimize}
          aria-label="Minimize"
        >
          <svg width="12" height="12" viewBox="0 0 12 12">
            <rect x="0" y="5" width="12" height="2" fill="currentColor" />
          </svg>
        </button>
        <button
          style={buttonStyle}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#2a2d2e')}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
          onClick={handleMaximize}
          aria-label="Maximize"
        >
          <svg width="12" height="12" viewBox="0 0 12 12">
            <rect x="1" y="1" width="10" height="10" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
        </button>
        <button
          style={{ ...buttonStyle, ...closeButtonStyle }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#e81123';
            e.currentTarget.style.color = '#ffffff';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = '#cccccc';
          }}
          onClick={handleClose}
          aria-label="Close"
        >
          <svg width="12" height="12" viewBox="0 0 12 12">
            <path
              d="M 1,1 L 11,11 M 11,1 L 1,11"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </div>
    </div>
  );
};

const titleBarStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  height: '32px',
  minHeight: '32px',
  maxHeight: '32px',
  flexShrink: 0,
  backgroundColor: '#2d2d30',
  borderBottom: '1px solid #3e3e42',
  userSelect: 'none',
  WebkitUserSelect: 'none',
  position: 'relative',
  zIndex: 1000,
  width: '100%',
} as React.CSSProperties & { WebkitAppRegion?: string };

const titleStyle: React.CSSProperties = {
  fontSize: '13px',
  fontWeight: 500,
  color: '#cccccc',
  paddingLeft: '12px',
  letterSpacing: '0.2px',
};

const buttonsContainerStyle: React.CSSProperties = {
  display: 'flex',
  height: '100%',
} as React.CSSProperties & { WebkitAppRegion?: string };

const buttonStyle: React.CSSProperties = {
  width: '46px',
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  border: 'none',
  backgroundColor: 'transparent',
  color: '#cccccc',
  cursor: 'pointer',
  transition: 'background-color 0.15s ease',
  outline: 'none',
};

const closeButtonStyle: React.CSSProperties = {
  transition: 'background-color 0.15s ease, color 0.15s ease',
};
