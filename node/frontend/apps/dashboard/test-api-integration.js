#!/usr/bin/env node

/**
 * API Integration Test Script
 * Tests the connection between frontend services and backend APIs
 */

const chalk = require('chalk');

// Mock fetch for Node.js environment
if (!global.fetch) {
  global.fetch = require('node-fetch');
}

const API_BASE_URL = 'http://localhost:8080';
const WS_BASE_URL = 'ws://localhost:8080';

// Color helpers
const success = chalk.green;
const error = chalk.red;
const warning = chalk.yellow;
const info = chalk.blue;
const dim = chalk.gray;

// Test results
let passedTests = 0;
let failedTests = 0;
const testResults = [];

async function testEndpoint(name, url, method = 'GET', body = null) {
  process.stdout.write(dim(`Testing ${name}... `));
  
  try {
    const options = {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
    };
    
    if (body) {
      options.body = JSON.stringify(body);
    }
    
    const response = await fetch(url, options);
    const data = await response.json().catch(() => null);
    
    if (response.ok) {
      console.log(success('✓'));
      passedTests++;
      testResults.push({
        name,
        status: 'passed',
        statusCode: response.status,
        data: data ? JSON.stringify(data).substring(0, 100) + '...' : 'No data',
      });
      return { success: true, data, status: response.status };
    } else {
      console.log(error('✗'));
      failedTests++;
      testResults.push({
        name,
        status: 'failed',
        statusCode: response.status,
        error: data?.error || response.statusText,
      });
      return { success: false, status: response.status, error: data?.error || response.statusText };
    }
  } catch (err) {
    console.log(error('✗'));
    failedTests++;
    testResults.push({
      name,
      status: 'error',
      error: err.message,
    });
    return { success: false, error: err.message };
  }
}

async function testWebSocket(name, endpoint) {
  return new Promise((resolve) => {
    process.stdout.write(dim(`Testing WebSocket ${name}... `));
    
    try {
      const WebSocket = require('ws');
      const ws = new WebSocket(`${WS_BASE_URL}${endpoint}`);
      
      const timeout = setTimeout(() => {
        ws.close();
        console.log(error('✗ (timeout)'));
        failedTests++;
        testResults.push({
          name: `WebSocket: ${name}`,
          status: 'failed',
          error: 'Connection timeout',
        });
        resolve({ success: false, error: 'Connection timeout' });
      }, 5000);
      
      ws.on('open', () => {
        clearTimeout(timeout);
        console.log(success('✓'));
        passedTests++;
        testResults.push({
          name: `WebSocket: ${name}`,
          status: 'passed',
        });
        ws.close();
        resolve({ success: true });
      });
      
      ws.on('error', (err) => {
        clearTimeout(timeout);
        console.log(error('✗'));
        failedTests++;
        testResults.push({
          name: `WebSocket: ${name}`,
          status: 'error',
          error: err.message,
        });
        resolve({ success: false, error: err.message });
      });
    } catch (err) {
      console.log(error('✗'));
      failedTests++;
      testResults.push({
        name: `WebSocket: ${name}`,
        status: 'error',
        error: err.message,
      });
      resolve({ success: false, error: err.message });
    }
  });
}

async function runTests() {
  console.log(info('\n🚀 Testing API Integration\n'));
  console.log(dim(`API Base URL: ${API_BASE_URL}`));
  console.log(dim(`WebSocket URL: ${WS_BASE_URL}\n`));
  
  // Test Node Metrics endpoints
  console.log(info('📊 Node Metrics API:'));
  await testEndpoint('GET /api/node/metrics', `${API_BASE_URL}/api/node/metrics`);
  await testEndpoint('GET /api/node/health', `${API_BASE_URL}/api/node/health`);
  await testEndpoint('GET /api/node/status', `${API_BASE_URL}/api/node/status`);
  await testEndpoint('GET /api/node/config', `${API_BASE_URL}/api/node/config`);
  
  // Test Scrapper endpoints
  console.log(info('\n📦 Scrapper API:'));
  await testEndpoint('GET /api/scrapper/datasets', `${API_BASE_URL}/api/scrapper/datasets`);
  await testEndpoint('GET /api/scrapper/models', `${API_BASE_URL}/api/scrapper/models`);
  await testEndpoint('GET /api/scrapper/config', `${API_BASE_URL}/api/scrapper/config`);
  await testEndpoint('GET /api/scrapper/datasets/stats', `${API_BASE_URL}/api/scrapper/datasets/stats`);
  await testEndpoint('GET /api/scrapper/models/stats', `${API_BASE_URL}/api/scrapper/models/stats`);
  
  // Test WebSocket endpoints
  console.log(info('\n🔌 WebSocket Connections:'));
  await testWebSocket('Node Metrics', '/ws/node-metrics');
  await testWebSocket('Scrapper Progress', '/ws/scrapper-progress');
  
  // Print summary
  console.log(info('\n📈 Test Summary:\n'));
  console.log(`  ${success('✓ Passed:')} ${passedTests}`);
  console.log(`  ${error('✗ Failed:')} ${failedTests}`);
  console.log(`  ${dim('Total:')} ${passedTests + failedTests}`);
  
  // Print detailed results for failed tests
  if (failedTests > 0) {
    console.log(error('\n❌ Failed Tests:'));
    testResults.filter(r => r.status !== 'passed').forEach(result => {
      console.log(`  ${error('•')} ${result.name}`);
      if (result.error) {
        console.log(`    ${dim('Error:')} ${result.error}`);
      }
      if (result.statusCode) {
        console.log(`    ${dim('Status:')} ${result.statusCode}`);
      }
    });
  }
  
  // Recommendations
  if (failedTests > 0) {
    console.log(warning('\n⚠️  Recommendations:'));
    console.log(dim('  1. Ensure the backend server is running on port 8080'));
    console.log(dim('  2. Check that all required endpoints are implemented'));
    console.log(dim('  3. Verify CORS is configured correctly for http://localhost:5173'));
    console.log(dim('  4. Check the backend logs for any errors'));
  } else {
    console.log(success('\n✨ All tests passed! The frontend is ready to connect to the backend.'));
  }
  
  process.exit(failedTests > 0 ? 1 : 0);
}

// Check if ws module is installed
try {
  require('ws');
} catch (err) {
  console.log(warning('\n⚠️  WebSocket module not found. Installing...'));
  require('child_process').execSync('npm install ws', { stdio: 'inherit' });
}

// Check if node-fetch is needed
if (!global.fetch) {
  try {
    require('node-fetch');
  } catch (err) {
    console.log(warning('\n⚠️  node-fetch module not found. Installing...'));
    require('child_process').execSync('npm install node-fetch@2', { stdio: 'inherit' });
  }
}

// Run tests
runTests().catch(err => {
  console.error(error('\n❌ Test script error:'), err);
  process.exit(1);
});