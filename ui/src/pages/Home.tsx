
import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { ArrowRight, Brain, Layers, Check } from 'lucide-react';

const Home = () => {
  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 -z-10" />
      
      {/* Hero section */}
      <main className="flex-1">
        <div className="container max-w-6xl mx-auto px-4 pt-8 pb-16">
          <motion.div 
            className="text-center space-y-4 mb-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <motion.div
              className="inline-block mb-2 px-3 py-1 bg-primary/10 text-primary rounded-full text-sm font-medium"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: 0.2 }}
            >
              AI-Powered Medical Imaging
            </motion.div>
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-4">
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-violet-600 to-indigo-600">
                Advanced Organ Segmentation
              </span>
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto">
              Analyze medical images with precision using our state-of-the-art AI model to automatically identify and segment organs.
            </p>
            
            <motion.div
              className="flex flex-wrap justify-center gap-4 mt-8"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.6 }}
            >
              <Link to="/segmentation">
                <motion.button
                  className="px-6 py-3 rounded-lg bg-primary text-white font-medium flex items-center gap-2 shadow-lg shadow-primary/20 hover:shadow-xl hover:shadow-primary/30 transition-all"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Try Segmentation
                  <ArrowRight className="h-4 w-4" />
                </motion.button>
              </Link>
              <Link to="/insights">
                <motion.button
                  className="px-6 py-3 rounded-lg bg-white text-primary font-medium flex items-center gap-2 border border-gray-200 shadow-sm hover:shadow-md hover:border-primary/20 transition-all"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.98 }}
                >
                  View Model Insights
                  <Layers className="h-4 w-4" />
                </motion.button>
              </Link>
            </motion.div>
          </motion.div>
          
          {/* Feature section */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto mt-16"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            {[
              {
                icon: <Brain className="h-8 w-8 text-purple-500" />,
                title: "AI-Powered Analysis",
                description: "Deep learning algorithms trained on thousands of medical scans for accurate organ identification."
              },
              {
                icon: <Layers className="h-8 w-8 text-blue-500" />,
                title: "Multi-Organ Segmentation",
                description: "Precisely segment up to 10 different organs and structures in abdominal CT scans."
              },
              {
                icon: <Check className="h-8 w-8 text-green-500" />,
                title: "Clinically Validated",
                description: "Validated against expert radiologist annotations with high accuracy metrics."
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                className="glass-card p-6 rounded-xl"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.2 + index * 0.1 }}
                whileHover={{ y: -5, transition: { duration: 0.2 } }}
              >
                <div className="rounded-full bg-white p-3 w-16 h-16 flex items-center justify-center mb-4 shadow-sm">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </motion.div>
            ))}
          </motion.div>
          
          {/* Demo image section */}
          <motion.div
            className="mt-20 max-w-5xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <h2 className="text-2xl md:text-3xl font-bold text-center mb-8">
              Advanced Organ Segmentation in Action
            </h2>
            <div className="glass-card rounded-xl p-6">
              <div className="aspect-video bg-gradient-to-r from-indigo-50 via-purple-50 to-pink-50 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <p className="text-muted-foreground mb-4">Interactive demo visualization</p>
                  <Link to="/segmentation">
                    <motion.button
                      className="px-4 py-2 rounded-lg bg-primary text-white font-medium flex items-center gap-2 mx-auto"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      Try it yourself
                      <ArrowRight className="h-4 w-4" />
                    </motion.button>
                  </Link>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </main>
      
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

export default Home;
