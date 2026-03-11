import { Activity, Target, TrendingUp } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card } from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

const accuracyData = [
  { model: "BERT", accuracy: 0.89, hits10: 0.92, mrr: 0.85 },
  { model: "RoBERTa", accuracy: 0.91, hits10: 0.94, mrr: 0.88 },
  { model: "GPT-3", accuracy: 0.87, hits10: 0.90, mrr: 0.83 },
  { model: "T5", accuracy: 0.93, hits10: 0.95, mrr: 0.9 },
  { model: "CLIP", accuracy: 0.85, hits10: 0.88, mrr: 0.81 },
];

const trainingData = [
  { epoch: 1, train: 0.65, val: 0.62 },
  { epoch: 2, train: 0.72, val: 0.7 },
  { epoch: 3, train: 0.78, val: 0.75 },
  { epoch: 4, train: 0.85, val: 0.82 },
  { epoch: 5, train: 0.89, val: 0.86 },
  { epoch: 6, train: 0.92, val: 0.88 },
];

export function ModelPerformance() {
  return (
    <div className="mx-auto max-w-[1440px] space-y-6 px-8 py-8">
      <div className="space-y-2">
        <h1>Model Performance Visualization</h1>
        <p className="text-muted-foreground">
          Comprehensive analysis of model performance across different metrics and datasets
        </p>
      </div>

      <Card className="p-4">
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-3">
            <label className="mb-2 block">Model</label>
            <Select defaultValue="all">
              <SelectTrigger>
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Models</SelectItem>
                <SelectItem value="bert">BERT</SelectItem>
                <SelectItem value="roberta">RoBERTa</SelectItem>
                <SelectItem value="gpt3">GPT-3</SelectItem>
                <SelectItem value="t5">T5</SelectItem>
                <SelectItem value="clip">CLIP</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="col-span-3">
            <label className="mb-2 block">Dataset</label>
            <Select defaultValue="amazon">
              <SelectTrigger>
                <SelectValue placeholder="Select dataset" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="amazon">Amazon Products</SelectItem>
                <SelectItem value="ebay">eBay Listings</SelectItem>
                <SelectItem value="aliexpress">AliExpress</SelectItem>
                <SelectItem value="custom">Custom Dataset</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="col-span-3">
            <label className="mb-2 block">Metric</label>
            <Select defaultValue="accuracy">
              <SelectTrigger>
                <SelectValue placeholder="Select metric" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="accuracy">Accuracy</SelectItem>
                <SelectItem value="hits">Hits@K</SelectItem>
                <SelectItem value="mrr">MRR</SelectItem>
                <SelectItem value="f1">F1 Score</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-12 gap-6">
        <Card className="col-span-4 p-6">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <p className="text-muted-foreground">Average Accuracy</p>
              <p className="text-3xl">89.2%</p>
              <div className="flex items-center gap-1 text-green-600">
                <TrendingUp className="h-4 w-4" />
                <span className="text-sm">+2.4% from baseline</span>
              </div>
            </div>
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <Target className="h-6 w-6 text-primary" />
            </div>
          </div>
        </Card>

        <Card className="col-span-4 p-6">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <p className="text-muted-foreground">Hits@10</p>
              <p className="text-3xl">91.8%</p>
              <div className="flex items-center gap-1 text-green-600">
                <TrendingUp className="h-4 w-4" />
                <span className="text-sm">+3.1% from baseline</span>
              </div>
            </div>
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-2/10">
              <Activity className="h-6 w-6 text-chart-2" />
            </div>
          </div>
        </Card>

        <Card className="col-span-4 p-6">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <p className="text-muted-foreground">Mean Reciprocal Rank</p>
              <p className="text-3xl">0.854</p>
              <div className="flex items-center gap-1 text-green-600">
                <TrendingUp className="h-4 w-4" />
                <span className="text-sm">+1.8% from baseline</span>
              </div>
            </div>
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-3/10">
              <TrendingUp className="h-6 w-6 text-chart-3" />
            </div>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-12 gap-6">
        <Card className="col-span-7 p-6">
          <div className="space-y-4">
            <div>
              <h3>Model Comparison: Accuracy &amp; Metrics</h3>
              <p className="text-sm text-muted-foreground">Performance across different models</p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={accuracyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="model" stroke="#6b7280" />
                <YAxis stroke="#6b7280" domain={[0, 1]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "white",
                    border: "1px solid #e5e7eb",
                    borderRadius: "0.5rem",
                  }}
                />
                <Legend />
                <Bar dataKey="accuracy" fill="#0891b2" name="Accuracy" radius={[4, 4, 0, 0]} />
                <Bar dataKey="hits10" fill="#06b6d4" name="Hits@10" radius={[4, 4, 0, 0]} />
                <Bar dataKey="mrr" fill="#6366f1" name="MRR" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="col-span-5 p-6">
          <div className="space-y-4">
            <div>
              <h3>Training Progress</h3>
              <p className="text-sm text-muted-foreground">Training vs validation accuracy</p>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="epoch"
                  stroke="#6b7280"
                  label={{ value: "Epoch", position: "insideBottom", offset: -5 }}
                />
                <YAxis stroke="#6b7280" domain={[0.5, 1]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "white",
                    border: "1px solid #e5e7eb",
                    borderRadius: "0.5rem",
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="train"
                  stroke="#0891b2"
                  strokeWidth={2}
                  name="Training"
                  dot={{ r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="val"
                  stroke="#ec4899"
                  strokeWidth={2}
                  name="Validation"
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card className="p-6">
        <h3 className="mb-4">Performance Summary</h3>
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-8">
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b py-2">
                <span>Best Performing Model</span>
                <span>T5 (93% accuracy)</span>
              </div>
              <div className="flex items-center justify-between border-b py-2">
                <span>Average Training Time</span>
                <span>2.4 hours per epoch</span>
              </div>
              <div className="flex items-center justify-between border-b py-2">
                <span>Dataset Size</span>
                <span>1.2M product entries</span>
              </div>
              <div className="flex items-center justify-between py-2">
                <span>Last Updated</span>
                <span>October 21, 2025</span>
              </div>
            </div>
          </div>
          <div className="col-span-4 rounded-lg bg-muted/30 p-4">
            <h4 className="mb-3">Key Insights</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>- T5 shows best overall performance</li>
              <li>- Minimal overfitting across models</li>
              <li>- MRR strongly correlates with accuracy</li>
              <li>- Vision models lag behind text models</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}
