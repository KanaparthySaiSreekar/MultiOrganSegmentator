
import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, ArrowRight, Check, X, ArrowLeft } from 'lucide-react';
import { toast } from "sonner";
import { Link } from 'react-router-dom';
import OrganLegend from '@/components/OrganLegend';
import ImageCard from '@/components/ImageCard';

const Index = () => {
  const [imageURL, setImageURL] = useState<string>('');
  const [outputImageURL, setOutputImageURL] = useState<string>('');
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
  const selectedFile = e.target.files?.[0];
  if (!selectedFile) return;
  if (imageURL) URL.revokeObjectURL(imageURL);
  if (outputImageURL) URL.revokeObjectURL(outputImageURL);
  setFile(selectedFile);
  const newImageURL = URL.createObjectURL(selectedFile);
  setImageURL(newImageURL);
  setOutputImageURL('');

  toast.success("Image uploaded successfully", {
    description: "You can now process the image.",
    icon: <Check className="h-4 w-4" />,
  });
};

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      const imageURL = URL.createObjectURL(droppedFile);
      setImageURL(imageURL);
      setOutputImageURL('');
      
      toast.success("Image uploaded successfully", {
        description: "You can now process the image.",
        icon: <Check className="h-4 w-4" />,
      });
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleProcessImage = async () => {
    if (!file) {
      toast.error("No image to process", {
        description: "Please upload an image first.",
        icon: <X className="h-4 w-4" />,
      });
      return;
    }
    
    setIsProcessing(true);
    const formData = new FormData();
    formData.append('image', file);
    
    try {
      const response = await fetch('http://localhost:8000/process_image', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Error processing image:', errorData.error);
        toast.error("Processing failed", {
          description: errorData.error || "An unknown error occurred.",
          icon: <X className="h-4 w-4" />,
        });
        return;
      }
      
      const blob = await response.blob();
      const outputImageURL = URL.createObjectURL(blob);
      setOutputImageURL(outputImageURL);
      
      toast.success("Image processed successfully", {
        description: "Your segmented image is ready.",
        icon: <Check className="h-4 w-4" />,
      });
    } catch (error) {
      console.error('Error processing image:', error);
      toast.error("Processing failed", {
        description: "Could not connect to the server. Please try again.",
        icon: <X className="h-4 w-4" />,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!outputImageURL) return;
    
    const link = document.createElement('a');
    link.href = outputImageURL;
    link.download = 'segmented-organ-image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    toast.success("Image downloaded", {
      description: "The segmented image has been saved to your device.",
      icon: <Check className="h-4 w-4" />,
    });
  };

  const handleReset = () => {
    setImageURL('');
    setOutputImageURL('');
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 -z-10" />
      
      {/* Content */}
      <motion.div 
        className="container max-w-6xl mx-auto px-4 py-8 md:py-16 flex-1 flex flex-col"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        {/* Header */}
        <motion.div 
          className="text-center mb-8 md:mb-16"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <Link to="/" className="inline-flex items-center gap-2 text-primary hover:underline mb-4">
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </Link>
          <motion.div
            className="inline-block mb-2 px-3 py-1 bg-primary/10 text-primary rounded-full text-sm font-medium"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            AI-Powered
          </motion.div>
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight mb-3">Organ Segmentation</h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Upload a medical image to automatically identify and color-code different organs.
          </p>
        </motion.div>
        
        {/* Main content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-5xl mx-auto w-full">
          {/* Input section */}
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.3 }}
            >
              <h2 className="text-xl font-semibold mb-4">Input Image</h2>
              
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
                ref={fileInputRef}
              />
              
              {!imageURL ? (
                <div
                  className="upload-zone h-64 md:h-80"
                  onClick={handleUploadClick}
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                >
                  <Upload className="h-10 w-10 mb-4 text-primary/60" />
                  <p className="font-medium">Drop your image here or click to browse</p>
                  <p className="text-sm text-muted-foreground mt-2">Supports JPG, PNG, DICOM</p>
                </div>
              ) : (
                <ImageCard
                  imageURL={imageURL}
                  title="Input Image"
                  onRetry={handleReset}
                />
              )}
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4, delay: 0.4 }}
              className="flex justify-center"
            >
              <button
                onClick={handleProcessImage}
                disabled={!file || isProcessing}
                className={`btn-primary px-6 py-2.5 rounded-lg flex items-center gap-2 ${
                  !file || isProcessing ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {isProcessing ? (
                  <>
                    <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    Process Image
                    <ArrowRight className="h-4 w-4" />
                  </>
                )}
              </button>
            </motion.div>
          </div>
          
          {/* Output section */}
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.5 }}
            >
              <h2 className="text-xl font-semibold mb-4">Segmentation Result</h2>
              
              <AnimatePresence mode="wait">
                {outputImageURL ? (
                  <ImageCard
                    key="output"
                    imageURL={outputImageURL}
                    title="Segmented Image"
                    onDownload={handleDownload}
                  />
                ) : (
                  <motion.div
                    key="placeholder"
                    className="glass-card rounded-xl overflow-hidden"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <div className="aspect-square w-full flex items-center justify-center bg-secondary/50">
                      <div className="text-center p-6">
                        <p className="text-muted-foreground">
                          {imageURL
                            ? "Process the image to see results"
                            : "Upload an image to get started"}
                        </p>
                      </div>
                    </div>
                    <div className="p-4">
                      <h3 className="font-medium text-center">Segmented Image</h3>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        </div>
        
        {/* Legend section */}
        <motion.div
          className="mt-10 md:mt-16"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.6 }}
        >
          <OrganLegend />
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

export default Index;
