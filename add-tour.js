const fs = require('fs');

const kgPath = 'C:\\CODE\\automation\\ai_explainer\\.understand-anything\\knowledge-graph.json';
const kg = JSON.parse(fs.readFileSync(kgPath, 'utf8'));

// Only add if not present
if (!kg.tour || kg.tour.length === 0) {
  const topNodes = kg.nodes.slice(0, 3).map(n => n.id);
  kg.tour = [
    {
      order: 1,
      title: "Project Start",
      description: "This is the start of the project.",
      nodeIds: topNodes
    }
  ];
  fs.writeFileSync(kgPath, JSON.stringify(kg, null, 2));
  console.log('Added a default tour.');
} else {
  console.log('Tour already exists.');
}
