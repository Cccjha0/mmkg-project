import { useState, useRef, useEffect, useMemo } from "react";
import { Card } from "./ui/card";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Search, Languages, X } from "lucide-react";
import ForceGraph3D from "react-force-graph-3d";
import SpriteText from "three-spritetext";

const DEFAULT_QUERY = "Pants";

export function KnowledgeGraphExplorer() {
  const [selectedNode, setSelectedNode] = useState(null);
  const [clickedNode, setClickedNode] = useState(null);
  const [showDetailDialog, setShowDetailDialog] = useState(false);
  const [clickedNodeConnections, setClickedNodeConnections] = useState([]);

  const [graphData, setGraphData] = useState({
    nodes: [],
    links: [],
  });

  const [query, setQuery] = useState(DEFAULT_QUERY);

  const [k, setK] = useState(1);
  const [n, setN] = useState(1);
  const [p, setP] = useState(1);

  const [language, setLanguage] = useState("en");

  const timerRef = useRef(null);

  // ForceGraph ref
  const fgRef = useRef();

  // graph 容器 ref
  const containerRef = useRef(null);

  // graph 尺寸
  const [graphSize, setGraphSize] = useState({
    width: 800,
    height: 600,
  });

  useEffect(() => {
    plot(DEFAULT_QUERY, k, n, p, language);
  }, []);

  const clampNumber = (value, min, max) => {
    if (!Number.isFinite(value)) return min;
    return Math.min(max, Math.max(min, value));
  };

  // =========================================
  // 自动读取 graph 容器尺寸
  // =========================================
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setGraphSize({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        });
      }
    };

    updateSize();

    window.addEventListener("resize", updateSize);

    return () => {
      window.removeEventListener("resize", updateSize);
    };
  }, []);

  // =========================================
  // 监听右侧面板内容变化，更新graph尺寸
  // =========================================
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setGraphSize({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        });
      }
    };

    updateSize();

    const resizeObserver = new ResizeObserver(updateSize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, [selectedNode, graphData.links.length]);

  // =========================================
  // graph 数据更新后自动居中（只在首次加载时）
  // =========================================
  useEffect(() => {
    if (graphData.nodes.length && fgRef.current) {
      setTimeout(() => {
        fgRef.current.zoomToFit(1000, 60);
      }, 500);
    }
  }, [graphData.nodes.length > 0]);

  // =========================================
  // 延迟搜索
  // =========================================
  const delayedPlot = (q, kVal, nVal, pVal, lang) => {
    clearTimeout(timerRef.current);

    timerRef.current = setTimeout(() => {
      if (q?.trim()) {
        plot(q, kVal, nVal, pVal, lang);
      }
    }, 700);
  };

  // =========================================
  // 请求后端
  // =========================================
  const plot = (q, kVal, nVal, pVal, lang = language) => {
    fetch(
      `http://127.0.0.1:5000/search/${kVal}/${nVal}/${pVal}/${encodeURIComponent(q)}?lang=${lang}`
    )
      .then((res) => res.json())
      .then((data) => {
        setGraphData(data);

        if (data?.nodes?.length) {
          setSelectedNode(data.nodes[0]);
          setClickedNode(null);
        }
      })
      .catch((err) => {
        console.log("Waiting for backend...");
      });
  };

  // =========================================
  // 获取节点标签（根据语言）
  // =========================================
  const getNodeLabel = (node) => {
    if (!node) return "";

    if (language === "zh") {
      return node.label_zh || node.label || node.id;
    } else {
      return node.label_en || node.label || node.id;
    }
  };

  // =========================================
  // 获取关系标签（根据语言）
  // =========================================
  const getRelationLabel = (link) => {
    if (!link) return "";

    if (language === "zh") {
      return link.relation_zh || link.relation || "";
    } else {
      return link.relation_en || link.relation || "";
    }
  };

  // =========================================
  // 切换语言
  // =========================================
  const toggleLanguage = () => {
    const newLang = language === "en" ? "zh" : "en";
    setLanguage(newLang);

    if (query.trim()) {
      plot(query, k, n, p, newLang);
    }
  };

  // =========================================
  // 判断是否为中心节点
  // =========================================
  const isCenterNode = (nodeId) => {
    return selectedNode && selectedNode.id === nodeId;
  };

    // =========================================
  // 获取连接节点（基于当前搜索结果子图）
  // =========================================
  const getConnectedNodes = (node) => {
    if (!node) return [];

    const connected = graphData.links
      .filter((l) => {
        // source 和 target 可能是对象或字符串
        const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
        const targetId = typeof l.target === 'object' ? l.target.id : l.target;

        return sourceId === node.id || targetId === node.id;
      })
      .map((link) => {
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;

        const otherNodeId = sourceId === node.id ? targetId : sourceId;

        const otherNode = graphData.nodes.find(
          (n) => n.id === otherNodeId
        );

        return otherNode ? { node: otherNode, relation: link } : null;
      })
      .filter(Boolean);

    return connected;
  };

  // =========================================
  // 点击节点 - 只显示子图中的连接
  // =========================================
  const handleNodeClick = (node) => {
    if (node) {
      setClickedNode(node);

      // 直接使用当前子图中的连接数据
      const subgraphConnections = getConnectedNodes(node);

      setClickedNodeConnections(subgraphConnections);
      setShowDetailDialog(true);
    }
  };

  // =========================================
  // 关闭弹窗
  // =========================================
  const closeDialog = () => {
    setShowDetailDialog(false);
    setClickedNode(null);
    setClickedNodeConnections([]);
  };

  // =========================================
  // 获取节点度数
  // =========================================
  const getNodeDegree = (node) => {
    if (!node) return 0;
    return clickedNodeConnections.length;
  };

  const selectedNodeConnections = useMemo(
    () => getConnectedNodes(selectedNode),
    [selectedNode, graphData.links, graphData.nodes]
  );

  return (
    <div className="max-w-[1600px] mx-auto px-6 py-6 space-y-4">
      {/* ========================================= */}
      {/* Header */}
      {/* ========================================= */}
      <div className="space-y-1">
        <h1 className="text-2xl font-bold">
          Knowledge Graph Explorer
        </h1>

        <p className="text-muted-foreground text-sm">
          Interactive 3D Knowledge Graph Visualization
        </p>
      </div>

      {/* ========================================= */}
      {/* Top Row: Search Panel + Statistics */}
      {/* ========================================= */}
      <div className="grid grid-cols-12 gap-4">
        {/* Search Panel - col-span-8 与图对齐 */}
        <div className="col-span-8">
          <Card className="p-4 h-[120px] flex flex-col justify-center">
            {/* 第一行：Search + Language */}
            <div className="flex items-end gap-3 mb-3">
              {/* Search - 固定较长宽度 */}
              <div className="flex-1">
                <label className="block text-xs mb-1.5 text-muted-foreground font-medium">
                  Search
                </label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    aria-label="Search"
                    placeholder={language === "en" ? "Search Product..." : "搜索产品..."}
                    className="pl-10 h-9"
                    value={query}
                    onChange={(e) => {
                      setQuery(e.target.value);
                      delayedPlot(e.target.value, k, n, p, language);
                    }}
                  />
                </div>
              </div>

              {/* Language Toggle */}
              <div className="w-[80px]">
                <label className="block text-xs mb-1.5 text-muted-foreground font-medium">
                  Lang
                </label>
                <Button
                  onClick={toggleLanguage}
                  variant="outline"
                  className="w-full h-9"
                >
                  <Languages className="w-4 h-4 mr-2" />
                  {language === "en" ? "EN" : "中文"}
                </Button>
              </div>


            </div>

            {/* 第二行：Top K + Neighbours + Prune - 平均分布 */}
            <div className="flex items-end gap-3">
              {/* Top K */}
              <div className="flex-1">
                <label className="block text-xs mb-1.5 text-muted-foreground font-medium">
                  Top K
                </label>
                <Input
                  aria-label="Top K"
                  type="number"
                  className="h-9"
                  min={1}
                  max={5}
                  value={k}
                  onChange={(e) => {
                    const value = clampNumber(Number(e.target.value), 1, 5);
                    setK(value);
                    delayedPlot(query, value, n, p, language);
                  }}
                />
              </div>

              {/* Neighbours */}
              <div className="flex-1">
                <label className="block text-xs mb-1.5 text-muted-foreground font-medium">
                  Neighbours
                </label>
                <Input
                  aria-label="Neighbours"
                  type="number"
                  className="h-9"
                  min={1}
                  max={2}
                  value={n}
                  onChange={(e) => {
                    const value = clampNumber(Number(e.target.value), 1, 2);
                    setN(value);
                    delayedPlot(query, k, value, p, language);
                  }}
                />
              </div>

              {/* Prune */}
              <div className="flex-1">
                <label className="block text-xs mb-1.5 text-muted-foreground font-medium">
                  Prune
                </label>
                <Input
                  aria-label="Prune"
                  type="number"
                  className="h-9"
                  min={1}
                  max={3}
                  value={p}
                  onChange={(e) => {
                    const value = clampNumber(Number(e.target.value), 1, 3);
                    setP(value);
                    delayedPlot(query, k, n, value, language);
                  }}
                />
              </div>
            </div>
          </Card>
        </div>

        {/* Graph Statistics - col-span-4 与右侧面板对齐 */}
        <div className="col-span-4">
          <Card className="p-4 h-[120px] flex flex-col justify-center">
            <h3 className="mb-3 font-semibold text-base">
              Graph Statistics
            </h3>

            {/* Total Nodes 和 Total Edges 纵向排列 */}
            <div className="space-y-2">
              <div className="flex justify-between items-center p-2 bg-slate-50 rounded">
                <span className="text-xs text-muted-foreground">Total Nodes</span>
                <span className="font-mono text-sm font-semibold">
                  {graphData.nodes.length}
                </span>
              </div>

              <div className="flex justify-between items-center p-2 bg-slate-50 rounded">
                <span className="text-xs text-muted-foreground">Total Edges</span>
                <span className="font-mono text-sm font-semibold">
                  {graphData.links.length}
                </span>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* ========================================= */}
      {/* Main Layout: Graph + Right Panel */}
      {/* ========================================= */}
      <div className="grid grid-cols-12 gap-4 items-stretch" style={{ minHeight: '600px' }}>
        {/* ========================================= */}
        {/* LEFT : 3D GRAPH */}
        {/* ========================================= */}
        <div className="col-span-8 flex flex-col">
          <Card className="overflow-hidden flex flex-col w-full h-full">
            {/* 3D Graph Header - 调宽上下边界 */}
            <div className="px-6 py-6 bg-gradient-to-r from-indigo-50 via-purple-50 to-pink-50 border-b border-indigo-100 flex-shrink-0">
              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <h3 className="font-bold text-lg text-gray-800">3D Knowledge Graph</h3>
                  <p className="text-xs text-muted-foreground mt-1">Interactive visualization • Scroll to zoom • Drag to rotate • Click node to view details</p>
                </div>
              </div>
            </div>

            {/* Graph Container */}
            <div
              ref={containerRef}
              className="flex-1 min-h-[500px]"
              style={{
                position: "relative",
                overflow: "hidden",
                background: "#FFFFFF",
              }}
            >
              {graphSize.height > 0 && (
                <ForceGraph3D
                  ref={fgRef}
                  width={graphSize.width}
                  height={graphSize.height}
                  graphData={graphData}
                  backgroundColor="#FFFFFF"

                  // physics
                  warmupTicks={100}
                  cooldownTicks={200}
                  cooldownTime={500}

                  // nodes
                  nodeRelSize={6}
                  nodeOpacity={1}
                  nodeAutoColorBy="group"

                  enableNodeDrag={false}
                  enableZoomInteraction={true}
                  enablePanInteraction={true}

                  // links
                  linkOpacity={0.4}
                  linkWidth={0.7}
                  linkColor={() => "#808080"}

                  // particles
                  linkDirectionalParticles={1}
                  linkDirectionalParticleWidth={0.6}
                  linkDirectionalParticleResolution={8}

                  // relation label
                  linkThreeObjectExtend={true}
                  linkThreeObject={(link) => {
                    const sprite = new SpriteText(getRelationLabel(link));
                    sprite.color = "#A9A9A9";
                    sprite.textHeight = 2;
                    sprite.fontFace = "Arial";
                    return sprite;
                  }}

                  // relation position
                  linkPositionUpdate={(sprite, { start, end }) => {
                    const middlePos = {
                      x: start.x + (end.x - start.x) / 2,
                      y: start.y + (end.y - start.y) / 2,
                      z: start.z + (end.z - start.z) / 2,
                    };
                    Object.assign(sprite.position, middlePos);
                  }}

                  // node label
                  nodeThreeObject={(node) => {
                    const label = getNodeLabel(node);
                    const sprite = new SpriteText(label);
                    sprite.color = node.color || "#222222";
                    sprite.textHeight = isCenterNode(node.id) ? 5 : 4;
                    sprite.fontFace = "Arial";
                    return sprite;
                  }}

                  // 自动居中
                  onEngineStop={() => {}}

                  // click
                  onNodeClick={(node) => {
                    handleNodeClick(node);
                  }}
                />
              )}
            </div>
          </Card>
        </div>

        {/* ========================================= */}
        {/* RIGHT PANEL - 不滚动，自然高度 */}
        {/* ========================================= */}
        <div className="col-span-4 flex flex-col">
          <div className="space-y-4 w-full">
            {/* Selected Node Details - 搜索后固定显示，点击其他节点不变化 */}
            {selectedNode && (
              <Card className="p-4 border-l-4 border-l-blue-500">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold text-base">
                    Selected Node
                  </h3>
                  <Badge variant="secondary" className="text-xs">
                    Search Result
                  </Badge>
                </div>

                <div className="grid grid-cols-2 gap-4">


                  {/* Right Column: Basic Information */}
                  <div className="space-y-4">
                    <div className="pb-2 ">
                      <span className="text-xs text-muted-foreground font-medium block mb-1.5">
                        Label ({language === "en" ? "EN" : "中文"})
                      </span>
                      <p className="text-sm font-medium leading-relaxed text-gray-900">
                        {getNodeLabel(selectedNode)}
                      </p>
                    </div>

                    <div className="pb-2">
                      <span className="text-xs text-muted-foreground font-medium block mb-1.5">
                        中文名称
                      </span>
                      <p className="text-sm text-gray-700 leading-relaxed">
                        {selectedNode.label_zh || "N/A"}
                      </p>
                    </div>

                    <div className="pb-2">
                      <span className="text-xs text-muted-foreground font-medium block mb-1.5">
                        English Name
                      </span>
                      <p className="text-sm text-gray-700 leading-relaxed">
                        {selectedNode.label_en || "N/A"}
                      </p>
                    </div>

                    <div className="pb-2">
                      <span className="text-xs text-muted-foreground font-medium block mb-1.5">
                        Entity ID
                      </span>
                      <p className="text-sm text-gray-700 leading-relaxed">
                        {selectedNode.id}
                      </p>
                    </div>
                  </div>
                  {/* Left Column: Image */}
                  <div>
                    <span className="text-xs text-muted-foreground font-medium block mb-2">
                      Image
                    </span>
                    <div className="w-full h-40 bg-slate-100 rounded-lg overflow-hidden flex items-center justify-center">
                      {selectedNode.metadata?.image ? (
                        <img
                          src={`http://127.0.0.1:5000${selectedNode.metadata.image}`}
                          alt={getNodeLabel(selectedNode)}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            e.currentTarget.style.display = 'none';
                            const sibling = e.currentTarget.nextElementSibling;
                            if (sibling) {
                              sibling.style.display = 'flex';
                            }
                          }}
                          onLoad={(e) => {
                            e.currentTarget.style.display = 'block';
                            const sibling = e.currentTarget.nextElementSibling;
                            if (sibling) {
                              sibling.style.display = 'none';
                            }
                          }}
                        />
                      ) : null}
                      <div
                        className="w-full h-full flex items-center justify-center text-muted-foreground text-xs text-center p-2"
                        style={{ display: selectedNode.metadata?.image ? 'none' : 'flex' }}
                      >
                        No image available
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            )}

            {/* Connected Nodes - 模仿参考图片样式 */}
            {selectedNode && (
              <Card className="p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-base">
                    Connected Nodes
                  </h3>
                  <Badge variant="outline" className="text-xs">
                    {selectedNodeConnections.length} connections
                  </Badge>
                </div>

                <div className="space-y-3 max-h-[350px] overflow-y-auto pr-1">
                  {selectedNodeConnections.length > 0 ? (
                    selectedNodeConnections.map(({ node, relation }, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between px-2 py-1.5 rounded-md hover:bg-gray-50 transition-colors cursor-pointer"
                        onClick={() => handleNodeClick(node)}
                      >
                        <div className="flex items-center gap-3 flex-1 min-w-0">
                          <div
                            className="w-3 h-3 rounded-full flex-shrink-0"
                            style={{ backgroundColor: node.color || '#888888' }}
                          ></div>
                          <span className="text-sm font-medium truncate text-gray-800">
                            {getNodeLabel(node)}
                          </span>
                        </div>
                        <Badge
                          variant="secondary"
                          className="text-xs px-3 py-0.5 ml-3 flex-shrink-0 bg-white border border-gray-200 text-gray-600 font-normal rounded-full hover:bg-gray-50 transition-colors"
                        >
                          {getRelationLabel(relation)}
                        </Badge>
                      </div>
                    ))
                  ) : (
                    <p className="text-xs text-muted-foreground italic text-center py-4">
                      No connections found
                    </p>
                  )}
                </div>
              </Card>
            )}
          </div>
        </div>
      </div>

      {/* ========================================= */}
      {/* Custom Detail Dialog - 点击节点弹出 */}
      {/* ========================================= */}
      {showDetailDialog && clickedNode && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={closeDialog}>
          <div
            className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto m-4"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Dialog Header */}
            <div className="sticky top-0 bg-white border-b px-6 py-4 flex items-center justify-between z-10">
              <div className="flex items-center gap-3">
                <h2 className="text-xl font-bold">Click Node Details</h2>
                <Badge variant="secondary">{getNodeLabel(clickedNode)}</Badge>
              </div>
              <button
                onClick={closeDialog}
                className="p-2 hover:bg-gray-100 rounded-full transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Dialog Content */}
            <div className="p-6 space-y-6">
              {/* Node Information - Two columns: Left=Info+Image, Right=Statistics */}

              <div className="grid grid-cols-2 gap-4">

                <Card className="p-4">
                  <h4 className="font-semibold text-sm mb-3 text-blue-600">Basic Information</h4>

                  <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-4">
                        <div className="pb-4 ">
                          <span className="text-xs text-muted-foreground">Current Label</span>
                          <p className="text-sm font-medium mt-0.5">{getNodeLabel(clickedNode)}</p>
                        </div>
                        <div className="pb-4 ">
                          <span className="text-xs text-muted-foreground">中文名称</span>
                          <p className="text-sm mt-0.5 text-gray-700">{clickedNode.label_zh || "N/A"}</p>
                        </div>
                        <div className="pb-4 ">
                          <span className="text-xs text-muted-foreground">English Name</span>
                          <p className="text-sm mt-0.5 text-gray-700">{clickedNode.label_en || "N/A"}</p>
                        </div>
                        <div className="pb-4 ">
                          <span className="text-xs text-muted-foreground">Entity ID</span>
                          <p className="text-xs font-mono text-gray-600 bg-gray-50 p-1.5 rounded mt-0.5 break-all">{clickedNode.id}</p>
                        </div>
                      </div>

                      {/* Image at the bottom of Basic Information */}
                      <div className="mt-3">
                        <span className="text-xs text-muted-foreground font-medium block mb-2">
                          Image
                        </span>
                        <div className="w-full h-20 bg-slate-100 rounded-lg overflow-hidden flex items-center justify-center">
                          {clickedNode.metadata?.image ? (
                            <img
                              src={`http://127.0.0.1:5000${clickedNode.metadata.image}`}
                              alt={getNodeLabel(clickedNode)}
                              className="w-full h-full object-cover"
                              onError={(e) => {
                                e.currentTarget.style.display = 'none';
                                const sibling = e.currentTarget.nextElementSibling;
                                if (sibling) {
                                  sibling.style.display = 'flex';
                                }
                              }}
                              onLoad={(e) => {
                                e.currentTarget.style.display = 'block';
                                const sibling = e.currentTarget.nextElementSibling;
                                if (sibling) {
                                  sibling.style.display = 'none';
                                }
                              }}
                            />
                          ) : null}
                          <div
                            className="w-full h-full flex items-center justify-center text-muted-foreground text-xs text-center p-2"
                            style={{ display: clickedNode.metadata?.image ? 'none' : 'flex' }}
                          >
                            No Image Available
                          </div>
                        </div>
                      </div>
                    </div>
                </Card>


                <Card className="p-4">
                  <h4 className="font-semibold text-sm mb-3 text-green-600">Statistics</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center p-2 bg-green-50 rounded">
                      <span className="text-xs text-muted-foreground">Connected Nodes</span>
                      <span className="font-mono text-lg font-bold text-green-700">
                        {clickedNodeConnections.length}
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-2 bg-slate-50 rounded">
                      <span className="text-xs text-muted-foreground">Connected Edges</span>
                      <span className="font-mono text-sm font-semibold">{clickedNodeConnections.length}</span>
                    </div>
                    <div className="flex justify-between items-center p-2 bg-slate-50 rounded">
                      <span className="text-xs text-muted-foreground">Total Graph Nodes</span>
                      <span className="font-mono text-sm font-semibold">{graphData.nodes.length}</span>
                    </div>
                    <div className="flex justify-between items-center p-2 bg-slate-50 rounded">
                      <span className="text-xs text-muted-foreground">Total Graph Edges</span>
                      <span className="font-mono text-sm font-semibold">{graphData.links.length}</span>
                    </div>
                  </div>
                </Card>
              </div>

              {/* Connected Nodes - 显示子图中的连接 */}
              <Card className="p-4">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="font-semibold text-base text-gray-800">Connected Nodes (Subgraph)</h4>
                  <Badge variant="outline" className="text-xs">{clickedNodeConnections.length} connections</Badge>
                </div>
                <div className="grid grid-cols-2 gap-6 max-h-[300px] overflow-y-auto">
                  {/* 左列 */}
                  <div className="space-y-3 pr-6 border-r border-dashed border-gray-300">
                    {clickedNodeConnections.length > 0 ? (
                      clickedNodeConnections
                        .slice(0, Math.ceil(clickedNodeConnections.length / 2))
                        .map(({ node, relation }, i) => (
                          <div
                            key={i}
                            className="flex items-center justify-between py-1.5 hover:bg-gray-50 rounded-md px-2 -mx-2 transition-colors"
                          >
                            <div className="flex items-center gap-3 flex-1 min-w-0">
                              <div
                                className="w-3 h-3 rounded-full flex-shrink-0"
                                style={{ backgroundColor: node.color || '#888888' }}
                              ></div>
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-gray-800 truncate">
                                  {getNodeLabel(node)}
                                </p>
                                <p className="text-xs text-gray-500 font-mono truncate mt-0.5">
                                  {node.id}
                                </p>
                              </div>
                            </div>
                            <Badge
                              variant="secondary"
                              className="text-xs px-3 py-0.5 ml-3 flex-shrink-0 bg-white border border-gray-200 text-gray-600 font-normal rounded-full"
                            >
                              {getRelationLabel(relation)}
                            </Badge>
                          </div>
                        ))
                    ) : (
                      <p className="text-xs text-muted-foreground italic text-center py-4">
                        No connections found
                      </p>
                    )}
                  </div>

                  {/* 右列 */}
                  <div className="space-y-3 pl-6">
                    {clickedNodeConnections.length > 0 ? (
                      clickedNodeConnections
                        .slice(Math.ceil(clickedNodeConnections.length / 2))
                        .map(({ node, relation }, i) => (
                          <div
                            key={i}
                            className="flex items-center justify-between py-1.5 hover:bg-gray-50 rounded-md px-2 -mx-2 transition-colors"
                          >
                            <div className="flex items-center gap-3 flex-1 min-w-0">
                              <div
                                className="w-3 h-3 rounded-full flex-shrink-0"
                                style={{ backgroundColor: node.color || '#888888' }}
                              ></div>
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-gray-800 truncate">
                                  {getNodeLabel(node)}
                                </p>
                                <p className="text-xs text-gray-500 font-mono truncate mt-0.5">
                                  {node.id}
                                </p>
                              </div>
                            </div>
                            <Badge
                              variant="secondary"
                              className="text-xs px-3 py-0.5 ml-3 flex-shrink-0 bg-white border border-gray-200 text-gray-600 font-normal rounded-full"
                            >
                              {getRelationLabel(relation)}
                            </Badge>
                          </div>
                        ))
                    ) : null}
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
