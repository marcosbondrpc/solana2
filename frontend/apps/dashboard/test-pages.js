#!/usr/bin/env node

const pages = [
  '/',
  '/dashboard',
  '/mev',
  '/arbitrage',
  '/jito',
  '/analytics',
  '/monitoring',
  '/settings'
];

async function testPage(url) {
  try {
    const response = await fetch(`http://localhost:3001${url}`);
    const html = await response.text();
    
    // Check if we get valid HTML with the app root
    const hasRoot = html.includes('id="root"');
    const hasVite = html.includes('vite');
    
    if (response.ok && hasRoot && hasVite) {
      console.log(`✅ ${url} - OK (Status: ${response.status})`);
      return true;
    } else {
      console.log(`❌ ${url} - Failed (Status: ${response.status}, hasRoot: ${hasRoot})`);
      return false;
    }
  } catch (error) {
    console.log(`❌ ${url} - Error: ${error.message}`);
    return false;
  }
}

async function testAll() {
  console.log('Testing all frontend pages...\n');
  
  let allPassed = true;
  for (const page of pages) {
    const passed = await testPage(page);
    if (!passed) allPassed = false;
  }
  
  console.log('\n' + (allPassed ? '✅ All pages are working!' : '❌ Some pages failed'));
  process.exit(allPassed ? 0 : 1);
}

// Wait a bit for server to be ready
setTimeout(testAll, 2000);