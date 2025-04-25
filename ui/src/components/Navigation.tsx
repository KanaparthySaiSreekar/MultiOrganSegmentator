
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Brain } from 'lucide-react';

const Navigation = () => {
  const location = useLocation();
  
  // Don't show navigation on the home page
  if (location.pathname === '/') {
    return null;
  }
  
  return (
    <motion.header
      className="border-b border-border/40 backdrop-blur-sm bg-white/70 sticky top-0 z-50"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="container max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <motion.div
            className="p-1.5 bg-primary/10 rounded-md"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Brain className="h-5 w-5 text-primary" />
          </motion.div>
          <span className="font-semibold text-lg">Organ Segmentation</span>
        </Link>
        
        <nav>
          <ul className="flex items-center gap-6">
            <li>
              <Link 
                to="/segmentation" 
                className={`text-sm transition-colors hover:text-primary ${
                  location.pathname === '/segmentation' ? 'text-primary font-medium' : 'text-muted-foreground'
                }`}
              >
                Segmentation
              </Link>
            </li>
            <li>
              <Link 
                to="/insights" 
                className={`text-sm transition-colors hover:text-primary ${
                  location.pathname === '/insights' ? 'text-primary font-medium' : 'text-muted-foreground'
                }`}
              >
                Model Insights
              </Link>
            </li>
          </ul>
        </nav>
      </div>
    </motion.header>
  );
};

export default Navigation;
