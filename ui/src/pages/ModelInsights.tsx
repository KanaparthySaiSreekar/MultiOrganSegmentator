
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { ArrowLeft, ChevronDown, Zap, TrendingUp, Award, BarChart3 } from 'lucide-react';
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

// Sample training data
const trainingData = [
  { epoch: 1, loss: 0.82, accuracy: 0.51, val_loss: 0.79, val_accuracy: 0.53 },
  { epoch: 5, loss: 0.56, accuracy: 0.68, val_loss: 0.54, val_accuracy: 0.71 },
  { epoch: 10, loss: 0.41, accuracy: 0.78, val_loss: 0.43, val_accuracy: 0.76 },
  { epoch: 15, loss: 0.32, accuracy: 0.82, val_loss: 0.39, val_accuracy: 0.79 },
  { epoch: 20, loss: 0.28, accuracy: 0.86, val_loss: 0.35, val_accuracy: 0.81 },
  { epoch: 25, loss: 0.25, accuracy: 0.88, val_loss: 0.32, val_accuracy: 0.83 },
  { epoch: 30, loss: 0.21, accuracy: 0.90, val_loss: 0.29, val_accuracy: 0.85 },
  { epoch: 35, loss: 0.19, accuracy: 0.91, val_loss: 0.27, val_accuracy: 0.86 },
  { epoch: 40, loss: 0.17, accuracy: 0.92, val_loss: 0.25, val_accuracy: 0.87 },
  { epoch: 45, loss: 0.16, accuracy: 0.93, val_loss: 0.24, val_accuracy: 0.88 },
  { epoch: 50, loss: 0.15, accuracy: 0.94, val_loss: 0.23, val_accuracy: 0.89 },
];

// Sample organ-specific accuracy data
const organAccuracyData = [
  { organ: 'Spleen', dice: 0.96, precision: 0.95, recall: 0.97 },
  { organ: 'Right Kidney', dice: 0.92, precision: 0.91, recall: 0.94 },
  { organ: 'Left Kidney', dice: 0.91, precision: 0.90, recall: 0.93 },
  { organ: 'Liver', dice: 0.95, precision: 0.94, recall: 0.96 },
  { organ: 'Gallbladder', dice: 0.87, precision: 0.85, recall: 0.89 },
  { organ: 'Stomach', dice: 0.84, precision: 0.82, recall: 0.86 },
  { organ: 'Aorta', dice: 0.93, precision: 0.92, recall: 0.94 },
  { organ: 'Inferior Vena Cava', dice: 0.89, precision: 0.88, recall: 0.91 },
  { organ: 'Portal Vein', dice: 0.83, precision: 0.81, recall: 0.85 },
  { organ: 'Pancreas', dice: 0.81, precision: 0.79, recall: 0.83 },
];

// Sample model architecture data for radar chart
const modelParamsData = [
  { subject: 'Params (M)', A: 24, fullMark: 30 },
  { subject: 'Training Time (h)', A: 18, fullMark: 30 },
  { subject: 'Inference Time (ms)', A: 8.5, fullMark: 10 },
  { subject: 'Epochs', A: 50, fullMark: 100 },
  { subject: 'Batch Size', A: 8, fullMark: 16 },
  { subject: 'Learning Rate (×10⁻⁴)', A: 3, fullMark: 10 },
];

// Accordion component for model details
const Accordion = ({ title, children }: { title: string, children: React.ReactNode }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <motion.div 
      className="glass-card rounded-xl overflow-hidden mb-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <motion.button
        className="w-full p-4 text-left font-medium flex justify-between items-center"
        onClick={() => setIsOpen(!isOpen)}
      >
        {title}
        <ChevronDown 
          className={`h-5 w-5 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} 
        />
      </motion.button>
      <motion.div
        initial={{ height: 0, opacity: 0 }}
        animate={{ 
          height: isOpen ? 'auto' : 0,
          opacity: isOpen ? 1 : 0
        }}
        transition={{ duration: 0.3 }}
        className="overflow-hidden"
      >
        <div className="p-4 pt-0 border-t">{children}</div>
      </motion.div>
    </motion.div>
  );
};

const ModelInsights = () => {
  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-50 via-purple-50 to-blue-50 -z-10" />
      
      {/* Content */}
      <motion.div 
        className="container max-w-6xl mx-auto px-4 py-8 md:py-16 flex-1 flex flex-col"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        {/* Header with back button */}
        <motion.div 
          className="mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <Link to="/" className="inline-flex items-center gap-2 text-primary hover:underline mb-4">
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </Link>
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight mb-3">Model Insights & Performance</h1>
          <p className="text-muted-foreground max-w-2xl">
            Detailed analysis of our organ segmentation model, including training metrics, performance statistics, and architectural insights.
          </p>
        </motion.div>
        
        {/* Key metrics section */}
        <motion.div
          className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          {[
            {
              icon: <Zap className="h-5 w-5 text-yellow-500" />,
              title: "24M",
              subtitle: "Model Parameters",
            },
            {
              icon: <TrendingUp className="h-5 w-5 text-green-500" />,
              title: "94%",
              subtitle: "Overall Accuracy",
            },
            {
              icon: <Award className="h-5 w-5 text-blue-500" />,
              title: "0.91",
              subtitle: "Average Dice Score",
            },
            {
              icon: <BarChart3 className="h-5 w-5 text-purple-500" />,
              title: "10",
              subtitle: "Organs Detected",
            },
          ].map((metric, index) => (
            <motion.div
              key={index}
              className="glass-card rounded-xl p-4 flex items-center gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.1 + index * 0.1 }}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
            >
              <div className="rounded-full bg-white p-2 flex items-center justify-center shadow-sm">
                {metric.icon}
              </div>
              <div>
                <h3 className="text-2xl font-bold">{metric.title}</h3>
                <p className="text-sm text-muted-foreground">{metric.subtitle}</p>
              </div>
            </motion.div>
          ))}
        </motion.div>
        
        {/* Training progress chart */}
        <motion.div
          className="mb-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <h2 className="text-2xl font-bold mb-4">Training Progress</h2>
          <div className="glass-card rounded-xl p-6">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trainingData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="epoch" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="accuracy" stroke="#8884d8" strokeWidth={2} />
                <Line type="monotone" dataKey="val_accuracy" stroke="#82ca9d" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
            <div className="flex items-center justify-center gap-6 mt-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-[#8884d8] rounded-full"></div>
                <span className="text-sm">Training Accuracy</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-[#82ca9d] rounded-full"></div>
                <span className="text-sm">Validation Accuracy</span>
              </div>
            </div>
          </div>
        </motion.div>
        
        {/* Per-organ performance */}
        <motion.div
          className="mb-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <h2 className="text-2xl font-bold mb-4">Organ-Specific Performance</h2>
          <div className="glass-card rounded-xl p-6">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={organAccuracyData.slice(0, 6)} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="organ" />
                <YAxis domain={[0.7, 1]} />
                <Tooltip />
                <Bar dataKey="dice" fill="#8884d8" name="Dice Score" />
                <Bar dataKey="precision" fill="#82ca9d" name="Precision" />
                <Bar dataKey="recall" fill="#ffc658" name="Recall" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
        
        {/* Model architecture and parameters */}
        <motion.div
          className="mb-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <h2 className="text-2xl font-bold mb-4">Model Architecture</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="glass-card rounded-xl p-6">
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={modelParamsData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="subject" />
                  <PolarRadiusAxis />
                  <Radar name="Model" dataKey="A" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div className="glass-card rounded-xl p-6">
              <h3 className="text-xl font-semibold mb-4">Architecture Details</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Base Architecture</span>
                  <span className="font-medium">U-Net with ResNet34 Encoder</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Loss Function</span>
                  <span className="font-medium">Combined Dice + Cross Entropy</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Optimizer</span>
                  <span className="font-medium">Adam</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Learning Rate</span>
                  <span className="font-medium">3 × 10⁻⁴ with reduction on plateau</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Data Augmentation</span>
                  <span className="font-medium">Rotation, Scaling, Flipping</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Training Dataset</span>
                  <span className="font-medium">850 annotated CT scans</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Test Dataset</span>
                  <span className="font-medium">150 independent CT scans</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
        
        {/* Detailed model information accordions */}
        <motion.div
          className="mb-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <h2 className="text-2xl font-bold mb-4">Detailed Information</h2>
          <Accordion title="Training Methodology">
            <p className="mb-4">
              Our organ segmentation model was trained using a multi-stage approach to maximize performance across all target organs:
            </p>
            <ol className="list-decimal pl-5 space-y-2 mb-4">
              <li>Pre-training on general medical imaging dataset to learn basic anatomical features</li>
              <li>Fine-tuning on organ-specific datasets with class weighting to handle imbalanced data</li>
              <li>Post-processing with anatomical constraints to ensure biologically plausible segmentations</li>
            </ol>
            <p>
              Training was performed on 4 NVIDIA A100 GPUs with a distributed training strategy to handle the large 3D volumes efficiently. We employed mixed precision training to accelerate the process without sacrificing accuracy.
            </p>
          </Accordion>
          
          <Accordion title="Performance Analysis">
            <p className="mb-4">
              The model achieves state-of-the-art performance across all evaluated organs, with particularly strong results for larger organs like the liver and spleen. Smaller structures like the pancreas and portal vein remain more challenging but still achieve clinical-grade accuracy.
            </p>
            <h4 className="font-semibold mb-2">Performance Highlights:</h4>
            <ul className="list-disc pl-5 space-y-1 mb-4">
              <li>Average Dice coefficient of 0.91 across all organs</li>
              <li>95th percentile inference time under 10ms on GPU</li>
              <li>Successful segmentation in 98.7% of test cases</li>
              <li>Robust performance across different CT scanners and protocols</li>
            </ul>
            <p>
              Performance was validated through 5-fold cross-validation on the training set and further verified on an independent test set collected from different medical centers to ensure generalizability.
            </p>
          </Accordion>
          
          <Accordion title="Technical Challenges & Solutions">
            <p className="mb-4">
              Several technical challenges were addressed during model development:
            </p>
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold">Challenge: Class Imbalance</h4>
                <p>
                  Solution: Combination of weighted loss functions, data augmentation focused on minority classes, and specialized sampling strategies during training.
                </p>
              </div>
              <div>
                <h4 className="font-semibold">Challenge: Boundary Precision</h4>
                <p>
                  Solution: Implementation of boundary-aware loss terms and deep supervision at multiple decoder levels to enhance edge detection.
                </p>
              </div>
              <div>
                <h4 className="font-semibold">Challenge: Inference Speed</h4>
                <p>
                  Solution: Model pruning, knowledge distillation from larger teacher models, and ONNX runtime optimization for deployment.
                </p>
              </div>
            </div>
          </Accordion>
        </motion.div>
        
        {/* Call to action */}
        <motion.div
          className="text-center mb-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.7 }}
        >
          <h2 className="text-2xl font-bold mb-4">Ready to Try It Yourself?</h2>
          <p className="text-muted-foreground max-w-xl mx-auto mb-6">
            Upload your own medical images and see the advanced organ segmentation AI in action.
          </p>
          <Link to="/segmentation">
            <motion.button
              className="px-6 py-3 rounded-lg bg-primary text-white font-medium flex items-center gap-2 mx-auto"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
            >
              Try Segmentation Now
              <ArrowLeft className="h-4 w-4 rotate-180" />
            </motion.button>
          </Link>
        </motion.div>
      </motion.div>
      
      {/* Footer */}
      <motion.div 
        className="py-4 text-center text-sm text-muted-foreground border-t border-border/40 mt-auto"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4, delay: 0.7 }}
      >
        <p>Organ Segmentation Tool • AI-Powered Medical Imaging</p>
      </motion.div>
    </div>
  );
};

export default ModelInsights;
