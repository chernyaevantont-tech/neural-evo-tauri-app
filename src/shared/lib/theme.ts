// Modern theme inspired by VS Code and Miro
export const theme = {
  colors: {
    // Background colors
    background: {
      primary: '#1e1e1e',
      secondary: '#252526',
      tertiary: '#2d2d30',
      canvas: '#1a1a1a',
      hover: '#2a2d2e',
    },
    // Border colors
    border: {
      primary: '#3e3e42',
      secondary: '#454545',
      focus: '#007acc',
      active: '#0e639c',
    },
    // Text colors
    text: {
      primary: '#cccccc',
      secondary: '#969696',
      tertiary: '#6a6a6a',
      accent: '#4fc3f7',
      success: '#4caf50',
      error: '#f44336',
      warning: '#ff9800',
    },
    // Node type colors (vibrant and modern)
    node: {
      input: '#6bcf7f',
      dense: '#4fc3f7',
      conv2d: '#ff9f43',
      pooling: '#ab47bc',
      flatten: '#7cb342',
      add: '#ef5350',
      concat: '#ec407a',
      output: '#ff5252',
    },
    // UI accent colors
    accent: {
      primary: '#007acc',
      secondary: '#0e639c',
      hover: '#1a8cd8',
      success: '#4caf50',
      error: '#f44336',
      warning: '#ff9800',
    },
  },
  // Typography
  typography: {
    fontFamily: "'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif",
    fontSize: {
      xs: '11px',
      sm: '12px',
      md: '13px',
      lg: '14px',
      xl: '16px',
      xxl: '18px',
    },
    fontWeight: {
      regular: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
  },
  // Spacing
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '12px',
    lg: '16px',
    xl: '20px',
    xxl: '24px',
  },
  // Border radius
  borderRadius: {
    sm: '2px',
    md: '4px',
    lg: '6px',
    xl: '8px',
  },
  // Shadows
  shadows: {
    sm: '0 1px 3px rgba(0, 0, 0, 0.3)',
    md: '0 2px 8px rgba(0, 0, 0, 0.4)',
    lg: '0 4px 16px rgba(0, 0, 0, 0.5)',
    focus: '0 0 0 2px rgba(0, 122, 204, 0.4)',
  },
  // Transitions
  transitions: {
    fast: '100ms ease',
    normal: '200ms ease',
    slow: '300ms ease',
  },
};

export type Theme = typeof theme;
