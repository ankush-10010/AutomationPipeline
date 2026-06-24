const fs = require('fs');

const kgPath = 'C:\\CODE\\automation\\ai_explainer\\.understand-anything\\knowledge-graph.json';
const kg = JSON.parse(fs.readFileSync(kgPath, 'utf8'));

const notebookId = 'file:notebooks/orchestrator_noImage_gpuVoice.ipynb';

// Find scripts to connect
const targets = [
  'file:scripts/scene_splitter.py',
  'file:scripts/clip_indexer.py',
  'file:scripts/clip_matcher.py',
  'file:scripts/assembler.py',
  'file:scripts/script_generator.py',
  'file:scripts/tts_local.py',
  'file:scripts/YOLO_inference.py'
];

let addedEdges = 0;

for (const target of targets) {
  // Check if edge already exists
  const exists = kg.edges.find(e => e.source === notebookId && e.target === target);
  if (!exists) {
    kg.edges.push({
      source: notebookId,
      target: target,
      type: 'calls',
      weight: 1.0,
      description: 'Notebook conceptually orchestrates this module'
    });
    addedEdges++;
  }
}

fs.writeFileSync(kgPath, JSON.stringify(kg, null, 2));
console.log('Added', addedEdges, 'semantic edges from notebook to scripts.');
