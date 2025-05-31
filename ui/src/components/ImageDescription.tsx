
import React from 'react';
import { motion } from 'framer-motion';
import { Loader2, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ImageDescriptionProps {
  imageURL: string | null;
  onGenerateDescription: () => void;
  description: string | null;
  isGenerating: boolean;
  title?: string;
}

const ImageDescription: React.FC<ImageDescriptionProps> = ({
  imageURL,
  onGenerateDescription,
  description,
  isGenerating,
  title = "Image Analysis"
}) => {
  if (!imageURL) return null;

  return (
    <motion.div
      className="glass-card rounded-xl overflow-hidden p-4 mt-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="flex items-center gap-2 mb-3">
        <Info className="h-4 w-4 text-primary" />
        <h3 className="font-medium">{title}</h3>
      </div>
      
      {!description && !isGenerating && (
        <div className="mb-4">
          <p className="text-sm text-muted-foreground mb-3">
            Generate an AI description to understand what this image contains.
          </p>
          <Button 
            onClick={onGenerateDescription}
            size="sm"
            className="flex items-center gap-2"
          >
            Generate Description
          </Button>
        </div>
      )}

      {isGenerating && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          <p className="text-sm">Generating description...</p>
        </div>
      )}

      {description && !isGenerating && (
        <motion.div 
          className="text-sm bg-secondary/20 rounded-lg p-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <p>{description}</p>
        </motion.div>
      )}
    </motion.div>
  );
};

export default ImageDescription;
