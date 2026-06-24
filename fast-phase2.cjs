const fs = require('fs');
const cp = require('child_process');
const path = require('path');

const projectRoot = 'C:\\CODE\\automation\\ai_explainer';
const pluginRoot = 'C:\\Users\\ankus\\.understand-anything\\repo\\understand-anything-plugin';
const tmpDir = path.join(projectRoot, '.understand-anything', 'tmp');
const intermediateDir = path.join(projectRoot, '.understand-anything', 'intermediate');

const batches = JSON.parse(fs.readFileSync(path.join(intermediateDir, 'batches.json'), 'utf-8'));

for (let i = 0; i < batches.batches.length; i++) {
  const batch = batches.batches[i];
  const inputPath = path.join(tmpDir, `ua-file-analyzer-input-${i}.json`);
  const extractPath = path.join(tmpDir, `ua-file-extract-results-${i}.json`);
  const outputPath = path.join(intermediateDir, `batch-${i}.json`);

  fs.writeFileSync(inputPath, JSON.stringify({
    projectRoot,
    batchFiles: batch.files,
    batchImportData: batch.batchImportData
  }));

  console.log(`Running extract-structure.mjs for batch ${i}...`);
  cp.execSync(`node "${path.join(pluginRoot, 'skills', 'understand', 'extract-structure.mjs')}" "${inputPath}" "${extractPath}"`);

  const extractResults = JSON.parse(fs.readFileSync(extractPath, 'utf-8'));
  const nodes = [];
  const edges = [];

  for (const res of extractResults.results) {
    const fileNodeId = `file:${res.path}`;
    nodes.push({
      id: fileNodeId,
      type: 'file',
      name: path.basename(res.path),
      filePath: res.path,
      summary: `File ${res.path} (${res.language})`,
      tags: [res.fileCategory],
      complexity: res.totalLines > 200 ? 'complex' : (res.totalLines > 50 ? 'moderate' : 'simple')
    });

    if (res.functions) {
      for (const fn of res.functions) {
        if (fn.endLine - fn.startLine < 5) continue;
        const fnNodeId = `function:${res.path}:${fn.name}`;
        nodes.push({
          id: fnNodeId,
          type: 'function',
          name: fn.name,
          filePath: res.path,
          lineRange: [fn.startLine, fn.endLine],
          summary: `Function ${fn.name}`,
          tags: ['function'],
          complexity: 'simple'
        });
        edges.push({
          source: fileNodeId,
          target: fnNodeId,
          type: 'contains',
          direction: 'forward',
          weight: 1.0
        });
      }
    }

    if (res.classes) {
      for (const cls of res.classes) {
        const clsNodeId = `class:${res.path}:${cls.name}`;
        nodes.push({
          id: clsNodeId,
          type: 'class',
          name: cls.name,
          filePath: res.path,
          lineRange: [cls.startLine, cls.endLine],
          summary: `Class ${cls.name}`,
          tags: ['class'],
          complexity: 'moderate'
        });
        edges.push({
          source: fileNodeId,
          target: clsNodeId,
          type: 'contains',
          direction: 'forward',
          weight: 1.0
        });
      }
    }

    const imports = batch.batchImportData[res.path] || [];
    for (const imp of imports) {
      edges.push({
        source: fileNodeId,
        target: `file:${imp}`,
        type: 'imports',
        direction: 'forward',
        weight: 0.7
      });
    }
  }

  fs.writeFileSync(outputPath, JSON.stringify({ nodes, edges }, null, 2));
}

console.log('All batches processed!');

