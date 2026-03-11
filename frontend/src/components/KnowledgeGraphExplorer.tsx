import { Card } from "./ui/card";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Search, Filter, RotateCcw, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";
import { Switch } from "./ui/switch";
import { Slider } from "./ui/slider";
import { useState } from "react";

// Mock data for graph nodes
const nodes = [
  { id: 1, label: "Running Shoes", type: "category", x: 300, y: 200, color: "#0891b2" },
  { id: 2, label: "Nike", type: "brand", x: 200, y: 100, color: "#6366f1" },
  { id: 3, label: "Athletic", type: "attribute", x: 400, y: 100, color: "#ec4899" },
  { id: 4, label: "Mesh", type: "material", x: 250, y: 300, color: "#8b5cf6" },
  { id: 5, label: "Black", type: "color", x: 350, y: 300, color: "#f59e0b" },
  { id: 6, label: "Product A", type: "product", x: 150, y: 200, color: "#10b981" },
  { id: 7, label: "Product B", type: "product", x: 450, y: 200, color: "#10b981" },
  { id: 8, label: "Lightweight", type: "attribute", x: 300, y: 400, color: "#ec4899" },
  { id: 9, label: "Adidas", type: "brand", x: 100, y: 250, color: "#6366f1" },
  { id: 10, label: "Sneakers", type: "category", x: 500, y: 250, color: "#0891b2" },
];

const edges = [
  { from: 1, to: 2 },
  { from: 1, to: 3 },
  { from: 1, to: 4 },
  { from: 1, to: 5 },
  { from: 6, to: 1 },
  { from: 7, to: 1 },
  { from: 6, to: 2 },
  { from: 7, to: 3 },
  { from: 1, to: 8 },
  { from: 9, to: 6 },
  { from: 10, to: 7 },
];

export function KnowledgeGraphExplorer() {
  const [selectedNode, setSelectedNode] = useState(nodes[0]);
  const [showCategories, setShowCategories] = useState(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState([0.8]);

  return (
    <div className="max-w-[1440px] mx-auto px-8 py-8 space-y-6">
      <div className="space-y-2">
        <h1>Knowledge Graph Explorer</h1>
        <p className="text-muted-foreground">
          Interactive visualization of commodity relationships and attributes
        </p>
      </div>

      {/* Search and Filters */}
      <Card className="p-4">
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-6">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search entities, products, or attributes..."
                className="pl-10 bg-input-background"
              />
            </div>
          </div>
          <div className="col-span-6 flex items-center justify-end gap-3">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-muted-foreground" />
              <Switch checked={showCategories} onCheckedChange={setShowCategories} />
              <span className="text-sm">Show Categories</span>
            </div>
            <Button variant="outline" size="sm">
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset View
            </Button>
          </div>
        </div>
        <div className="grid grid-cols-12 gap-4 mt-4">
          <div className="col-span-4 flex items-center gap-3">
            <span className="text-sm text-muted-foreground whitespace-nowrap">Confidence:</span>
            <Slider
              value={confidenceThreshold}
              onValueChange={setConfidenceThreshold}
              max={1}
              step={0.1}
              className="flex-1"
            />
            <span className="text-sm min-w-[3ch]">{confidenceThreshold[0].toFixed(1)}</span>
          </div>
        </div>
      </Card>

      {/* Main Content */}
      <div className="grid grid-cols-12 gap-6">
        {/* Visualization Panel */}
        <Card className="col-span-8 bg-slate-900 border-slate-800 overflow-hidden">
          <div className="p-4 border-b border-slate-800 flex items-center justify-between">
            <h3 className="text-white">Graph Visualization</h3>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" className="text-white hover:bg-slate-800">
                <ZoomIn className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="sm" className="text-white hover:bg-slate-800">
                <ZoomOut className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="sm" className="text-white hover:bg-slate-800">
                <Maximize2 className="w-4 h-4" />
              </Button>
            </div>
          </div>
          <div className="relative" style={{ height: '600px' }}>
            <svg width="100%" height="100%" className="bg-slate-900">
              {/* Draw edges */}
              {edges.map((edge, i) => {
                const fromNode = nodes.find(n => n.id === edge.from);
                const toNode = nodes.find(n => n.id === edge.to);
                if (!fromNode || !toNode) return null;
                return (
                  <line
                    key={i}
                    x1={fromNode.x}
                    y1={fromNode.y}
                    x2={toNode.x}
                    y2={toNode.y}
                    stroke="#06b6d4"
                    strokeWidth="3"
                    opacity="0.6"
                  />
                );
              })}
              
              {/* Draw nodes */}
              {nodes.map((node) => (
                <g key={node.id}>
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={node.id === selectedNode.id ? 30 : 25}
                    fill={node.color}
                    stroke={node.id === selectedNode.id ? "#ffffff" : node.color}
                    strokeWidth={node.id === selectedNode.id ? 3 : 0}
                    opacity="0.9"
                    className="cursor-pointer transition-all hover:opacity-100"
                    onClick={() => setSelectedNode(node)}
                  />
                  <text
                    x={node.x}
                    y={node.y + 45}
                    textAnchor="middle"
                    fill="#e2e8f0"
                    fontSize="12"
                    className="pointer-events-none"
                  >
                    {node.label}
                  </text>
                </g>
              ))}
            </svg>

            {/* Legend */}
            <div className="absolute bottom-4 left-4 bg-slate-800/90 backdrop-blur rounded-lg p-4 text-white">
              <div className="text-sm mb-3">Node Types</div>
              <div className="space-y-2 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#0891b2]"></div>
                  <span>Category</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#6366f1]"></div>
                  <span>Brand</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#10b981]"></div>
                  <span>Product</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#ec4899]"></div>
                  <span>Attribute</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#8b5cf6]"></div>
                  <span>Material</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#f59e0b]"></div>
                  <span>Color</span>
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Info Panel */}
        <div className="col-span-4 space-y-6">
          <Card className="p-6">
            <h3 className="mb-4">Node Details</h3>
            <div className="space-y-4">
              <div>
                <span className="text-sm text-muted-foreground">Selected Node</span>
                <p className="mt-1">{selectedNode.label}</p>
              </div>
              <div>
                <span className="text-sm text-muted-foreground">Type</span>
                <div className="mt-1">
                  <Badge style={{ backgroundColor: selectedNode.color, color: 'white' }}>
                    {selectedNode.type}
                  </Badge>
                </div>
              </div>
              <div>
                <span className="text-sm text-muted-foreground">Node ID</span>
                <p className="mt-1">{selectedNode.id}</p>
              </div>
              <div>
                <span className="text-sm text-muted-foreground">Connections</span>
                <p className="mt-1">
                  {edges.filter(e => e.from === selectedNode.id || e.to === selectedNode.id).length} edges
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="mb-4">Connected Nodes</h3>
            <div className="space-y-2">
              {edges
                .filter(e => e.from === selectedNode.id || e.to === selectedNode.id)
                .map((edge, i) => {
                  const connectedNode = nodes.find(
                    n => n.id === (edge.from === selectedNode.id ? edge.to : edge.from)
                  );
                  if (!connectedNode) return null;
                  return (
                    <div
                      key={i}
                      className="flex items-center justify-between p-2 rounded hover:bg-muted/50 cursor-pointer"
                      onClick={() => setSelectedNode(connectedNode)}
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: connectedNode.color }}
                        ></div>
                        <span className="text-sm">{connectedNode.label}</span>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {connectedNode.type}
                      </Badge>
                    </div>
                  );
                })}
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="mb-4">Graph Statistics</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Nodes</span>
                <span>{nodes.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Edges</span>
                <span>{edges.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Avg. Degree</span>
                <span>{(edges.length * 2 / nodes.length).toFixed(1)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Graph Density</span>
                <span>0.24</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}