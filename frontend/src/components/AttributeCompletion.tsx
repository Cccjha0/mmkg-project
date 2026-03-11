import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { ImageWithFallback } from "./figma/ImageWithFallback";
import { CheckCircle2, Edit3, Sparkles } from "lucide-react";
import { Badge } from "./ui/badge";

export function AttributeCompletion() {
  return (
    <div className="max-w-[1440px] mx-auto px-8 py-8 space-y-6">
      <div className="space-y-2">
        <h1>Attribute Auto-Completion</h1>
        <p className="text-muted-foreground">
          AI-powered product attribute prediction and completion
        </p>
      </div>

      <div className="grid grid-cols-12 gap-8">
        {/* Left Side - Product Image */}
        <div className="col-span-5">
          <Card className="p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3>Product Image</h3>
              <Badge className="bg-green-100 text-green-800 hover:bg-green-100">
                <Sparkles className="w-3 h-3 mr-1" />
                AI Analyzed
              </Badge>
            </div>
            <div className="aspect-square bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg overflow-hidden flex items-center justify-center">
              <ImageWithFallback
                src="https://images.unsplash.com/photo-1593548826648-0357b8c884a9?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxibGFjayUyMHJ1bm5pbmclMjBzaG9lcyUyMGNsb3NldXAlMjB3aGl0ZSUyMGJhY2tncm91bmR8ZW58MXx8fHwxNzYxMTI3OTkxfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
                alt="Black running shoes"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="space-y-2 pt-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Confidence Score</span>
                <span>94.5%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-primary h-2 rounded-full" style={{ width: '94.5%' }}></div>
              </div>
            </div>
          </Card>
        </div>

        {/* Right Side - Form Panel */}
        <div className="col-span-7">
          <Card className="p-6 space-y-6">
            <div>
              <h3>Product Details</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Review and confirm AI-predicted attributes
              </p>
            </div>

            {/* Product Description */}
            <div className="space-y-2">
              <label>Product Description</label>
              <Textarea
                placeholder="black athletic shoes, lightweight and breathable"
                defaultValue="Black athletic running shoes with breathable mesh upper, lightweight EVA midsole, and durable rubber outsole. Features cushioned insole for comfort during long runs."
                rows={4}
                className="resize-none bg-input-background"
              />
            </div>

            {/* Predicted Fields */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  Brand
                  <Badge variant="outline" className="text-xs">Predicted</Badge>
                </label>
                <Select defaultValue="nike">
                  <SelectTrigger className="bg-input-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="nike">Nike</SelectItem>
                    <SelectItem value="adidas">Adidas</SelectItem>
                    <SelectItem value="puma">Puma</SelectItem>
                    <SelectItem value="newbalance">New Balance</SelectItem>
                    <SelectItem value="asics">ASICS</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  Category
                  <Badge variant="outline" className="text-xs">Predicted</Badge>
                </label>
                <Select defaultValue="running">
                  <SelectTrigger className="bg-input-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="running">Running Shoes</SelectItem>
                    <SelectItem value="training">Training Shoes</SelectItem>
                    <SelectItem value="casual">Casual Sneakers</SelectItem>
                    <SelectItem value="basketball">Basketball Shoes</SelectItem>
                    <SelectItem value="hiking">Hiking Shoes</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  Material
                  <Badge variant="outline" className="text-xs">Predicted</Badge>
                </label>
                <Select defaultValue="mesh">
                  <SelectTrigger className="bg-input-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mesh">Mesh</SelectItem>
                    <SelectItem value="leather">Leather</SelectItem>
                    <SelectItem value="synthetic">Synthetic</SelectItem>
                    <SelectItem value="canvas">Canvas</SelectItem>
                    <SelectItem value="knit">Knit</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  Color
                  <Badge variant="outline" className="text-xs">Predicted</Badge>
                </label>
                <Select defaultValue="black">
                  <SelectTrigger className="bg-input-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="black">Black</SelectItem>
                    <SelectItem value="white">White</SelectItem>
                    <SelectItem value="gray">Gray</SelectItem>
                    <SelectItem value="blue">Blue</SelectItem>
                    <SelectItem value="red">Red</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  Gender
                  <Badge variant="outline" className="text-xs">Predicted</Badge>
                </label>
                <Select defaultValue="unisex">
                  <SelectTrigger className="bg-input-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="unisex">Unisex</SelectItem>
                    <SelectItem value="men">Men</SelectItem>
                    <SelectItem value="women">Women</SelectItem>
                    <SelectItem value="kids">Kids</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  Size Range
                  <Badge variant="outline" className="text-xs">Predicted</Badge>
                </label>
                <Select defaultValue="7-12">
                  <SelectTrigger className="bg-input-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="7-12">7-12 US</SelectItem>
                    <SelectItem value="6-11">6-11 US</SelectItem>
                    <SelectItem value="8-13">8-13 US</SelectItem>
                    <SelectItem value="5-10">5-10 US</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Additional Attributes */}
            <div className="bg-muted/30 rounded-lg p-4 space-y-2">
              <h4>Additional Attributes</h4>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">Lightweight</Badge>
                <Badge variant="secondary">Breathable</Badge>
                <Badge variant="secondary">Cushioned</Badge>
                <Badge variant="secondary">Athletic</Badge>
                <Badge variant="secondary">Outdoor</Badge>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3 pt-2">
              <Button className="flex-1 bg-primary hover:bg-primary/90">
                <CheckCircle2 className="w-4 h-4 mr-2" />
                Confirm Attributes
              </Button>
              <Button variant="outline" className="flex-1">
                <Edit3 className="w-4 h-4 mr-2" />
                Edit Manually
              </Button>
            </div>
          </Card>
        </div>
      </div>

      {/* Processing Info */}
      <Card className="p-4">
        <div className="grid grid-cols-12 gap-4 text-sm">
          <div className="col-span-3">
            <span className="text-muted-foreground">Processing Time:</span> 0.43s
          </div>
          <div className="col-span-3">
            <span className="text-muted-foreground">Model Used:</span> CLIP + T5
          </div>
          <div className="col-span-3">
            <span className="text-muted-foreground">Attributes Predicted:</span> 6/6
          </div>
          <div className="col-span-3">
            <span className="text-muted-foreground">Avg. Confidence:</span> 91.2%
          </div>
        </div>
      </Card>
    </div>
  );
}