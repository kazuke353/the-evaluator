const fs = require('fs');
const { execSync } = require('child_process');
const crypto = require('crypto');
const path = require('path');

// Read input from stdin
let input = '';
process.stdin.on('data', (chunk) => {
  input += chunk;
});

process.stdin.on('end', async () => {
  try {
    // Parse the input JSON
    const data = JSON.parse(input);
    const { code, tests } = data;
    
    if (!code || !tests) {
      throw new Error('Missing required fields: code and tests');
    }

    // Create a temporary directory for the execution
    const tempDir = path.join('/tmp', crypto.randomBytes(16).toString('hex'));
    fs.mkdirSync(tempDir, { recursive: true });

    // Write the code to a temporary file
    const codeFile = path.join(tempDir, 'solution.ts');
    fs.writeFileSync(codeFile, code);

    // Write the test file
    const testFile = path.join(tempDir, 'test.ts');
    
    // Create a test file that imports the solution and runs the tests
    const testCode = `
import { expect } from 'chai';
import * as solution from './solution';

${tests}

// Run the tests and report results
const testResults = [];

// Helper function to capture test results
function runTest(name, testFn) {
  try {
    testFn();
    testResults.push({ name, passed: true });
  } catch (error) {
    testResults.push({ name, passed: false, error: error.message });
  }
}

// Execute all tests
${generateTestCalls(tests)}

// Output the results as JSON
console.log(JSON.stringify({ results: testResults }));
`;

    fs.writeFileSync(testFile, testCode);

    // Execute the tests
    try {
      const output = execSync(`cd ${tempDir} && ts-node test.ts`, {
        timeout: 5000, // 5 second timeout
        encoding: 'utf-8'
      });
      
      // Parse and return the test results
      process.stdout.write(output);
    } catch (error) {
      // Handle execution errors
      const errorOutput = {
        error: true,
        message: error.message,
        stdout: error.stdout ? error.stdout.toString() : '',
        stderr: error.stderr ? error.stderr.toString() : ''
      };
      process.stdout.write(JSON.stringify(errorOutput));
    }

    // Clean up temporary files
    try {
      fs.unlinkSync(codeFile);
      fs.unlinkSync(testFile);
      fs.rmdirSync(tempDir);
    } catch (cleanupError) {
      // Ignore cleanup errors
    }
  } catch (error) {
    // Handle JSON parsing or other errors
    process.stdout.write(JSON.stringify({
      error: true,
      message: error.message
    }));
  }
});

// Helper function to generate test calls from the test code
function generateTestCalls(testCode) {
  // Simple regex to extract test function names
  // This is a basic implementation and might need refinement
  const testRegex = /function\s+test(\w+)\s*\(\)/g;
  const matches = testCode.matchAll(testRegex);
  
  let testCalls = '';
  for (const match of matches) {
    if (match[1]) {
      testCalls += `runTest('test${match[1]}', test${match[1]});\n`;
    }
  }
  
  // If no tests were found with the regex, add a fallback
  if (!testCalls) {
    testCalls = `
    // Fallback: try to run any exported test functions
    for (const key in global) {
      if (key.startsWith('test') && typeof global[key] === 'function') {
        runTest(key, global[key]);
      }
    }`;
  }
  
  return testCalls;
}
