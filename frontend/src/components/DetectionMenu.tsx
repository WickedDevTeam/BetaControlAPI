import React, { useState } from 'react';
import { ChevronDownIcon } from '@heroicons/react/24/outline';

export interface DetectionArea {
  id: string;
  label: string;
  description: string;
  category: 'male' | 'female';
  type: 'covered' | 'uncovered' | 'general';
}

interface CategoryProps {
  title: string;
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
  darkMode: boolean;
}

interface DetectionMenuProps {
  enabled: Record<string, boolean>;
  onToggle: (id: string) => void;
  darkMode: boolean;
}

const Category: React.FC<CategoryProps> = ({ title, isOpen, onToggle, children, darkMode }) => (
  <div className={`mb-4 rounded-lg transition-all duration-200 ${
    darkMode ? 'bg-gray-900/50' : 'bg-gray-50'
  }`}>
    <button
      onClick={onToggle}
      className={`w-full px-4 py-3 flex justify-between items-center text-left rounded-lg transition-all duration-200 ${
        darkMode 
          ? 'bg-gray-900 hover:bg-gray-800' 
          : 'bg-gray-50 hover:bg-gray-100'
      }`}
    >
      <span className={`font-medium ${
        darkMode 
          ? 'bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent' 
          : ''
      }`}>
        {title}
      </span>
      <ChevronDownIcon
        className={`w-5 h-5 transform transition-transform duration-200 ${
          isOpen ? 'rotate-180' : ''
        } ${darkMode ? 'text-purple-400' : 'text-gray-500'}`}
      />
    </button>
    {isOpen && (
      <div className={`p-4 rounded-b-lg ${darkMode ? 'bg-gray-900/50' : 'bg-white'}`}>
        {children}
      </div>
    )}
  </div>
);

export const detectionAreas: DetectionArea[] = [
  // Face detection (using both NudeNet and dlib)
  { id: 'face', label: 'Face', description: 'Detect and censor faces (using NudeNet and dlib)', category: 'male', type: 'general' },
  { id: 'eyes', label: 'Eyes', description: 'Detect and censor eyes', category: 'female', type: 'general' },
  { id: 'mouth', label: 'Mouth', description: 'Detect and censor mouth', category: 'female', type: 'general' },
  
  // Body parts (using NudeNet's accurate detection)
  { id: 'exposed_breast_f', label: 'Exposed Female Breasts', description: 'Detect and censor exposed female breasts', category: 'female', type: 'uncovered' },
  { id: 'covered_breast_f', label: 'Covered Female Breasts', description: 'Detect and censor covered female breasts', category: 'female', type: 'covered' },
  { id: 'exposed_genitalia_f', label: 'Exposed Female Genitalia', description: 'Detect and censor exposed female genitalia', category: 'female', type: 'uncovered' },
  { id: 'covered_genitalia_f', label: 'Covered Female Genitalia', description: 'Detect and censor covered female genitalia', category: 'female', type: 'covered' },
  { id: 'exposed_buttocks', label: 'Exposed Buttocks', description: 'Detect and censor exposed buttocks', category: 'female', type: 'uncovered' },
  { id: 'covered_buttocks', label: 'Covered Buttocks', description: 'Detect and censor covered buttocks', category: 'female', type: 'covered' },
  
  // Male specific (using NudeNet)
  { id: 'exposed_breast_m', label: 'Exposed Male Chest', description: 'Detect and censor exposed male chest', category: 'male', type: 'uncovered' },
  { id: 'exposed_genitalia_m', label: 'Exposed Male Genitalia', description: 'Detect and censor exposed male genitalia', category: 'male', type: 'uncovered' },
  
  // Additional body parts
  { id: 'belly', label: 'Belly', description: 'Detect and censor belly area', category: 'female', type: 'general' },
  { id: 'feet', label: 'Feet', description: 'Detect and censor feet', category: 'female', type: 'general' },
];

const DetectionMenu: React.FC<DetectionMenuProps> = ({ enabled, onToggle, darkMode }) => {
  const [openCategories, setOpenCategories] = useState({
    male: true,
    female: true,
  });

  const toggleCategory = (category: 'male' | 'female') => {
    setOpenCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
  };

  const renderToggle = (area: DetectionArea) => (
    <div key={area.id} className="mb-3 last:mb-0">
      <div className="flex items-center group">
        <button
          onClick={() => onToggle(area.id)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 focus:outline-none ${
            enabled[area.id] ? 'bg-green-400' : 'bg-gray-400'
          }`}
          role="switch"
          aria-checked={enabled[area.id]}
        >
          <span
            className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform duration-200 ${
              enabled[area.id] ? 'translate-x-5' : 'translate-x-1'
            }`}
          />
        </button>
        <div className="ml-3">
          <div className={`text-sm font-medium transition-colors ${
            darkMode ? (enabled[area.id] ? 'text-green-400' : 'text-gray-300') : (enabled[area.id] ? 'text-green-600' : 'text-gray-900')
          }`}>
            {area.label}
          </div>
          <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
            {area.description}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-4">
      <Category
        title="Male Detection"
        isOpen={openCategories.male}
        onToggle={() => toggleCategory('male')}
        darkMode={darkMode}
      >
        <div className="space-y-3">
          {detectionAreas
            .filter(area => area.category === 'male')
            .map(renderToggle)}
        </div>
      </Category>

      <Category
        title="Female Detection"
        isOpen={openCategories.female}
        onToggle={() => toggleCategory('female')}
        darkMode={darkMode}
      >
        <div className="space-y-3">
          {detectionAreas
            .filter(area => area.category === 'female')
            .map(renderToggle)}
        </div>
      </Category>
    </div>
  );
};

export default DetectionMenu; 