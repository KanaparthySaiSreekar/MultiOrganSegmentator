import { motion } from 'framer-motion';

interface OrganColor {
  name: string;
  color: string;
}

const organColors: OrganColor[] = [
  { name: 'Spleen', color: '#8800B9' },
  { name: 'Right Kidney', color: '#00E5FF' },
  { name: 'Left Kidney', color: '#FF0000' },
  { name: 'Liver', color: '#FF00FF' },
  { name: 'Gallbladder', color: '#3FF63F' },
  { name: 'Stomach', color: '#BD8A43' },
  { name: 'Aorta', color: 'darkblue' },
  { name: 'Inferior Vena Cava', color: '#FF8C00' },
  { name: 'Portal Vein', color: 'darkgreen' },
  { name: 'Pancreas', color: '#EAEA08' },
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
