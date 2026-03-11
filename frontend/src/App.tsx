import { useState } from "react";
import { BarChart3, FileText, Network } from "lucide-react";

import { AttributeCompletion } from "./components/AttributeCompletion";
import { KnowledgeGraphExplorer } from "./components/KnowledgeGraphExplorer";
import { ModelPerformance } from "./components/ModelPerformance";

type Page = "performance" | "completion" | "graph";

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>("performance");

  const navItems = [
    {
      id: "performance" as Page,
      label: "Model Performance",
      icon: BarChart3,
    },
    {
      id: "completion" as Page,
      label: "Attribute Completion",
      icon: FileText,
    },
    {
      id: "graph" as Page,
      label: "Knowledge Graph",
      icon: Network,
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-50 border-b border-border bg-card shadow-sm">
        <div className="mx-auto max-w-[1440px] px-8">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                <Network className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h2 className="leading-tight">Multimodal Commodity</h2>
                <p className="text-xs text-muted-foreground">Knowledge Graph System</p>
              </div>
            </div>

            <nav className="flex items-center gap-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentPage === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setCurrentPage(item.id)}
                    className={`flex items-center gap-2 rounded-lg px-4 py-2 transition-colors ${
                      isActive
                        ? "bg-primary text-primary-foreground"
                        : "text-foreground hover:bg-muted"
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>
        </div>
      </header>

      <main>
        {currentPage === "performance" && <ModelPerformance />}
        {currentPage === "completion" && <AttributeCompletion />}
        {currentPage === "graph" && <KnowledgeGraphExplorer />}
      </main>

      <footer className="mt-12 border-t border-border">
        <div className="mx-auto max-w-[1440px] px-8 py-6">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <div>&copy; 2025 Multimodal Commodity Knowledge Graph System</div>
            <div className="flex items-center gap-6">
              <span>Version 1.0.0</span>
              <span>Last updated: October 21, 2025</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
