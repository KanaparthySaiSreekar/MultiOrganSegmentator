
import React from 'react';
import { motion } from 'framer-motion';
import { Download, RefreshCw } from 'lucide-react';

interface ImageCardProps {
  imageURL: string;
  title: string;
  onDownload?: () => void;
  onRetry?: () => void;
  showActions?: boolean;
}

const ImageCard: React.FC<ImageCardProps> = ({
  imageURL,
  title,
  onDownload,
  onRetry,
  showActions = true,
}) => {
  return (
    <motion.div 
      className="glass-card rounded-xl overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="relative">
        <div className="aspect-square w-full flex items-center justify-center bg-secondary/50 overflow-hidden">
          {imageURL ? (
            <motion.img 
              src={imageURL} 
              alt={title} 
              className="max-w-full max-h-full object-contain"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
            />
          ) : (
            <div className="flex items-center justify-center h-full w-full">
              <p className="text-muted-foreground">No image to display</p>
            </div>
          )}
        </div>
        
        {showActions && imageURL && (
          <motion.div 
            className="absolute bottom-4 right-4 flex space-x-2"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.3 }}
          >
            {onDownload && (
              <motion.button
                onClick={onDownload}
                className="p-2 rounded-full bg-white/90 backdrop-blur-sm shadow-md text-primary hover:bg-white transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                aria-label="Download segmented image"
              >
                <Download size={18} />
              </motion.button>
            )}
            
            {onRetry && (
              <motion.button 
                onClick={onRetry}
                className="p-2 rounded-full bg-white/90 backdrop-blur-sm shadow-md text-primary hover:bg-white transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                aria-label="Retry with a new image"
              >
                <RefreshCw size={18} />
              </motion.button>
            )}
          </motion.div>
        )}
      </div>
      
      <div className="p-4">
        <h3 className="font-medium text-center">{title}</h3>
      </div>
    </motion.div>
  );
};

export default ImageCard;
