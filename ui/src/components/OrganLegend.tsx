
import React from 'react';
import { motion } from 'framer-motion';

interface OrganColor {
  name: string;
  color: string;
}

const organColors: OrganColor[] = [
  { name: 'Spleen', color: '#FF0094' },
  { name: 'Right Kidney', color: '#FFFF00' },
  { name: 'Left Kidney', color: '#0000FF' },
  { name: 'Liver', color: '#FF00FF' },
  { name: 'Gallbladder', color: '#00FF00' },
  { name: 'Stomach', color: '#7497C6' },
  { name: 'Aorta', color: '#FF0000' },
  { name: 'Inferior Vena Cava', color: '#0992FF' },
  { name: 'Portal Vein', color: '#415B00' },
  { name: 'Pancreas', color: '#00FFFF' },
  { name: 'Background', color: '#000000' },
];

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05,
    },
  },
};

const item = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0 },
};

const OrganLegend = () => {
  return (
    <motion.div 
      className="glass-card p-6 rounded-xl w-full max-w-md mx-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
    >
      <h3 className="text-lg font-medium mb-4">Color Legend</h3>
      <motion.div 
        className="grid grid-cols-2 gap-2 sm:grid-cols-3"
        variants={container}
        initial="hidden"
        animate="show"
      >
        {organColors.map((organ) => (
          <motion.div 
            key={organ.name} 
            className="flex items-center gap-2 p-2"
            variants={item}
          >
            <span 
              className="w-4 h-4 rounded-full flex-shrink-0" 
              style={{ backgroundColor: organ.color }}
            />
            <span className="text-sm truncate">{organ.name}</span>
          </motion.div>
        ))}
      </motion.div>
    </motion.div>
  );
};

export default OrganLegend;
