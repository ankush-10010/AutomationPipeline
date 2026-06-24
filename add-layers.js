const fs = require('fs');

const kgPath = 'C:\\CODE\\automation\\ai_explainer\\.understand-anything\\knowledge-graph.json';
const kg = JSON.parse(fs.readFileSync(kgPath, 'utf8'));

const codeNodes = [];
const configNodes = [];
const docsNodes = [];
const dataNodes = [];
const functionNodes = [];
const otherNodes = [];

for (const n of kg.nodes) {
  if (n.type === 'function' || n.type === 'class') {
    functionNodes.push(n.id);
  } else if (n.type === 'config') {
    configNodes.push(n.id);
  } else if (n.type === 'document' || n.type === 'docs') {
    docsNodes.push(n.id);
  } else if (n.type === 'data') {
    dataNodes.push(n.id);
  } else if (n.type === 'file') {
    codeNodes.push(n.id);
  } else {
    otherNodes.push(n.id);
  }
}

kg.layers = [
  {
    id: 'layer:code',
    name: 'Application Code',
    description: 'Core application source files.',
    nodeIds: codeNodes
  },
  {
    id: 'layer:functions',
    name: 'Functions and Classes',
    description: 'Extracted structural components.',
    nodeIds: functionNodes
  },
  {
    id: 'layer:config',
    name: 'Configuration',
    description: 'Project configuration files.',
    nodeIds: [...configNodes, ...otherNodes]
  },
  {
    id: 'layer:docs',
    name: 'Documentation & Data',
    description: 'Documentation and data files.',
    nodeIds: [...docsNodes, ...dataNodes]
  }
];

// Clean up any empty layers
kg.layers = kg.layers.filter(l => l.nodeIds.length > 0);

fs.writeFileSync(kgPath, JSON.stringify(kg, null, 2));
console.log('Layers added:', kg.layers.map(l => l.name).join(', '));
