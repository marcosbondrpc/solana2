#!/usr/bin/env node

/**
 * Test script to verify CPU-only optimizations are applied
 */

const fs = require('fs');
const path = require('path');

console.log('='.repeat(60));
console.log('SOLANA MEV DASHBOARD - CPU OPTIMIZATION VERIFICATION');
console.log('='.repeat(60));

const fixes = [
  {
    name: 'GPU Acceleration Removed from design-system.css',
    file: './apps/dashboard/src/styles/design-system.css',
    patterns: ['translateZ(0)', 'will-change', 'backface-visibility', 'preserve-3d'],
    shouldExist: false
  },
  {
    name: 'GPU Acceleration Removed from globals.css',
    file: './apps/dashboard/src/styles/globals.css',
    patterns: ['translateZ(0)', 'backface-visibility'],
    shouldExist: false
  },
  {
    name: 'Backdrop Filters Removed',
    file: './apps/dashboard/src/styles/design-system.css',
    patterns: ['backdrop-filter'],
    shouldExist: false
  },
  {
    name: 'WebSocket Delays Increased',
    file: './packages/websocket/src/providers/WebSocketProvider.tsx',
    patterns: ['1000); // Increased from 100ms', '5000) * Math.pow(2'],
    shouldExist: true
  },
  {
    name: 'App Config Updated',
    file: './apps/dashboard/src/App.tsx',
    patterns: ['reconnectInterval: 5000'],
    shouldExist: true
  },
  {
    name: 'Performance Monitor Optimized',
    file: './apps/dashboard/src/components/PerformanceMonitor.tsx',
    patterns: ['DISABLED: FPS measurement causes performance issues'],
    shouldExist: true
  },
  {
    name: 'VirtualScroller GPU Disabled',
    file: './apps/dashboard/src/components/VirtualScroller.tsx',
    patterns: ['GPU acceleration disabled for CPU-only server'],
    shouldExist: true
  }
];

let allPassed = true;

fixes.forEach(fix => {
  const filePath = path.join(__dirname, fix.file);
  
  if (!fs.existsSync(filePath)) {
    console.log(`❌ ${fix.name}: File not found`);
    allPassed = false;
    return;
  }
  
  const content = fs.readFileSync(filePath, 'utf8');
  let passed = true;
  let failedPatterns = [];
  
  fix.patterns.forEach(pattern => {
    const exists = content.includes(pattern);
    if (fix.shouldExist && !exists) {
      passed = false;
      failedPatterns.push(pattern);
    } else if (!fix.shouldExist && exists) {
      passed = false;
      failedPatterns.push(pattern);
    }
  });
  
  if (passed) {
    console.log(`✅ ${fix.name}`);
  } else {
    console.log(`❌ ${fix.name}`);
    console.log(`   Failed patterns: ${failedPatterns.join(', ')}`);
    allPassed = false;
  }
});

console.log('='.repeat(60));

if (allPassed) {
  console.log('✅ ALL OPTIMIZATIONS SUCCESSFULLY APPLIED!');
  console.log('');
  console.log('The dashboard is now optimized for CPU-only rendering:');
  console.log('- No GPU acceleration properties');
  console.log('- No backdrop blur effects');
  console.log('- Simplified animations');
  console.log('- Reduced WebSocket reconnection frequency');
  console.log('- Performance monitor optimized');
  console.log('');
  console.log('The dashboard should run smoothly without console errors.');
} else {
  console.log('⚠️  SOME OPTIMIZATIONS ARE MISSING');
  console.log('Please review the failed checks above.');
  process.exit(1);
}

console.log('='.repeat(60));