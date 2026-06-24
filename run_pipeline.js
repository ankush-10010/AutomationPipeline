const fs = require('fs');
const cp = require('child_process');
const path = require('path');

const targetProject = 'C:\\CODE\\automation\\ai_explainer';
const pluginRoot = 'C:\\Users\\ankus\\.understand-anything\\repo\\understand-anything-plugin';
const skillDir = path.join(pluginRoot, 'skills', 'understand');
const outDir = path.join(targetProject, '.understand-anything');
const intermediateDir = path.join(outDir, 'intermediate');
const tmpDir = path.join(outDir, 'tmp');

if (!fs.existsSync(intermediateDir)) fs.mkdirSync(intermediateDir, { recursive: true });
if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });

console.log('1. Scanning project...');
const scanFilesPath = path.join(tmpDir, 'ua-scan-files.json');
cp.execSync(`node "${path.join(skillDir, 'scan-project.mjs')}" "${targetProject}" "${scanFilesPath}"`);

const fileData = JSON.parse(fs.readFileSync(scanFilesPath, 'utf8'));

console.log('2. Extracting imports...');
const importMapInputPath = path.join(tmpDir, 'ua-import-map-input.json');
const importMapOutputPath = path.join(tmpDir, 'ua-import-map-output.json');
fs.writeFileSync(importMapInputPath, JSON.stringify({
  projectRoot: targetProject,
  files: fileData.files
}));
cp.execSync(`node "${path.join(skillDir, 'extract-import-map.mjs')}" "${importMapInputPath}" "${importMapOutputPath}"`);

let importData = { importMap: {} };
if (fs.existsSync(importMapOutputPath)) {
  importData = JSON.parse(fs.readFileSync(importMapOutputPath, 'utf8'));
}

console.log('3. Assembling scan results...');
const scanResult = {
  version: '1.0',
  projectRoot: targetProject,
  analyzedAt: new Date().toISOString(),
  gitCommitHash: 'unknown',
  projectName: 'ai_explainer',
  projectDescription: 'VibeCodingMax codebase',
  languages: ['python', 'json', 'yaml'],
  frameworks: [],
  totalLines: fileData.totalLines,
  filteredByIgnore: fileData.filteredByIgnore,
  complexityEstimate: fileData.estimatedComplexity,
  files: fileData.files,
  importMap: importData.importMap
};
fs.writeFileSync(path.join(intermediateDir, 'scan-result.json'), JSON.stringify(scanResult, null, 2));

console.log('4. Computing batches...');
cp.execSync(`node "${path.join(skillDir, 'compute-batches.mjs')}" "${targetProject}"`);

console.log('5. Running fast Phase 2...');
const fastPhase2Content = fs.readFileSync('C:\\CODE\\Understand-Anything\\fast-phase2.cjs', 'utf-8')
  .replace(/C:\\\\CODE\\\\Understand-Anything/g, targetProject.replace(/\\/g, '\\\\'));
fs.writeFileSync('C:\\CODE\\automation\\ai_explainer\\fast-phase2.cjs', fastPhase2Content);
cp.execSync(`node "C:\\CODE\\automation\\ai_explainer\\fast-phase2.cjs"`);

console.log('6. Merging batch graphs...');
cp.execSync(`python "${path.join(skillDir, 'merge-batch-graphs.py')}" "${targetProject}"`);

console.log('7. Finalizing knowledge graph...');
const assembled = JSON.parse(fs.readFileSync(path.join(intermediateDir, 'assembled-graph.json'), 'utf8'));
const finalGraph = {
  version: '1.0.0',
  project: {
    name: 'ai_explainer',
    languages: scanResult.languages,
    frameworks: scanResult.frameworks,
    description: scanResult.projectDescription,
    analyzedAt: scanResult.analyzedAt,
    gitCommitHash: scanResult.gitCommitHash
  },
  nodes: assembled.nodes,
  edges: assembled.edges,
  layers: [],
  tour: []
};
fs.writeFileSync(path.join(outDir, 'knowledge-graph.json'), JSON.stringify(finalGraph, null, 2));

console.log('8. Building fingerprints...');
fs.writeFileSync(path.join(intermediateDir, 'fingerprint-input.json'), JSON.stringify({
  projectRoot: targetProject,
  sourceFilePaths: scanResult.files.map(f => f.path),
  gitCommitHash: scanResult.gitCommitHash
}));
cp.execSync(`node "${path.join(skillDir, 'build-fingerprints.mjs')}" "${path.join(intermediateDir, 'fingerprint-input.json')}"`);

console.log('9. Writing meta.json...');
fs.writeFileSync(path.join(outDir, 'meta.json'), JSON.stringify({
  lastAnalyzedAt: finalGraph.project.analyzedAt,
  gitCommitHash: finalGraph.project.gitCommitHash,
  fileCount: scanResult.files.length,
  mode: 'full'
}));

console.log('All Done!');
