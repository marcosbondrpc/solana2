export const theme = {
  colors: {
    // Cyberpunk-inspired color palette
    primary: '#00ff41', // Matrix green
    secondary: '#00ffff', // Cyan accent
    danger: '#ff0040', // Warning red
    warning: '#ffaa00', // Alert orange
    
    // Background layers
    bg: {
      primary: '#0a0a0f',
      secondary: '#12121a',
      tertiary: '#1a1a25',
      glass: 'rgba(18, 18, 26, 0.85)',
    },
    
    // Gradients
    gradients: {
      primary: 'linear-gradient(135deg, #00ff41 0%, #00ffff 100%)',
      danger: 'linear-gradient(135deg, #ff0040 0%, #ff6060 100%)',
      surface: 'linear-gradient(180deg, rgba(0,255,65,0.05) 0%, rgba(0,255,255,0.03) 100%)',
    },
    
    // Text
    text: {
      primary: '#ffffff',
      secondary: '#a0a0b8',
      muted: '#6a6a7e',
      accent: '#00ff41',
    },
    
    // Borders
    border: {
      primary: 'rgba(0, 255, 65, 0.3)',
      secondary: 'rgba(0, 255, 255, 0.2)',
      glass: 'rgba(255, 255, 255, 0.1)',
    },
  },
  
  effects: {
    glass: `
      background: rgba(18, 18, 26, 0.85);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 8px 32px rgba(0, 255, 65, 0.1);
    `,
    glow: {
      primary: '0 0 20px rgba(0, 255, 65, 0.5)',
      secondary: '0 0 20px rgba(0, 255, 255, 0.5)',
      danger: '0 0 20px rgba(255, 0, 64, 0.5)',
    },
    neon: {
      text: `
        text-shadow: 
          0 0 10px rgba(0, 255, 65, 0.8),
          0 0 20px rgba(0, 255, 65, 0.6),
          0 0 30px rgba(0, 255, 65, 0.4);
      `,
    },
  },
  
  animation: {
    fast: '150ms',
    normal: '300ms',
    slow: '500ms',
    spring: {
      type: 'spring',
      damping: 20,
      stiffness: 300,
    },
  },
  
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px',
    xxl: '48px',
  },
  
  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '12px',
    xl: '16px',
  },
  
  fontSize: {
    xs: '10px',
    sm: '12px',
    base: '14px',
    lg: '16px',
    xl: '20px',
    '2xl': '24px',
    '3xl': '32px',
  },
};

export type Theme = typeof theme;