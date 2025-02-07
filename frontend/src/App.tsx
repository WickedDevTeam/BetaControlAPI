import { useState, useCallback, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import { SunIcon, MoonIcon } from '@heroicons/react/24/outline'
import DetectionMenu, { detectionAreas, DetectionArea } from './components/DetectionMenu'
import './App.css'

type Effect = 'pixelation' | 'blur' | 'blackbox' | 'ruin' | 'sticker'

interface ProcessedImage {
  processed_image: string
  regions: Array<{
    type: string
    coords?: [number, number, number, number]
    polygon?: number[][]
    confidence?: number
    label?: string
  }>
}

// Enhanced logging utility
const Logger = {
  info: (message: string, data?: any) => {
    console.log(
      `%c${new Date().toISOString()} [INFO] ${message}`,
      'color: #00b894',
      data ? data : ''
    );
  },
  warn: (message: string, data?: any) => {
    console.warn(
      `%c${new Date().toISOString()} [WARN] ${message}`,
      'color: #fdcb6e',
      data ? data : ''
    );
  },
  error: (message: string, error?: any) => {
    console.error(
      `%c${new Date().toISOString()} [ERROR] ${message}`,
      'color: #d63031',
      error ? error : ''
    );
  },
  debug: (message: string, data?: any) => {
    console.debug(
      `%c${new Date().toISOString()} [DEBUG] ${message}`,
      'color: #0984e3',
      data ? data : ''
    );
  }
};

// Create an axios instance with default config
const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true
});

// Add request/response interceptors for better error handling
api.interceptors.request.use(
  (config) => {
    Logger.info(`🌐 Making request to: ${config.url}`, {
      method: config.method,
      headers: config.headers,
      data: config.data ? JSON.parse(JSON.stringify(config.data)) : undefined
    });
    return config;
  },
  (error) => {
    Logger.error('❌ Request configuration error:', error);
    return Promise.reject(error);
  }
);

api.interceptors.response.use(
  (response) => {
    Logger.info(`✅ Received successful response from: ${response.config.url}`, {
      status: response.status,
      statusText: response.statusText,
      data: response.data ? '(Response data available)' : '(No data)'
    });
    return response;
  },
  (error) => {
    if (error.code === 'ECONNABORTED') {
      Logger.error('⏱️ Request timeout', {
        url: error.config?.url,
        timeout: error.config?.timeout
      });
    } else if (!error.response) {
      Logger.error('🔌 Network error - no response', {
        url: error.config?.url,
        message: error.message
      });
    } else {
      Logger.error('🚫 Response error:', {
        url: error.config?.url,
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data
      });
    }
    return Promise.reject(error);
  }
);

// Add new interface for sticker categories
interface StickerCategory {
  id: string
  name: string
}

// Add sticker categories based on the folders we found
const STICKER_CATEGORIES: StickerCategory[] = [
  { id: 'Gay Labels', name: 'Gay Labels' },
  { id: 'Gay', name: 'Gay' },
  { id: 'Ass', name: 'Ass' },
  { id: 'Pig', name: 'Pig' },
  { id: 'All Emojis', name: 'All Emojis' },
  { id: 'Labels', name: 'Labels' }
]

// Helper function to determine if effect uses strength
const effectUsesStrength = (effect: Effect): boolean => {
  return effect === 'pixelation' || effect === 'blur'
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [effect, setEffect] = useState<Effect>('pixelation')
  const [strength, setStrength] = useState<number>(7)
  const [enabledAreas, setEnabledAreas] = useState<Record<string, boolean>>(() => {
    const initialState: Record<string, boolean> = {};
    detectionAreas.forEach(area => {
      initialState[area.id] = false;
    });
    return initialState;
  })
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stickerCategory, setStickerCategory] = useState<string>(STICKER_CATEGORIES[0].id)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result as string)
        setProcessedImage(null)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp']
    },
    multiple: false
  })

  const handleProcess = async () => {
    if (!file) return

    setIsProcessing(true)
    setError(null)

    try {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onloadend = async () => {
        const base64Data = reader.result?.toString().split(',')[1]

        const enabledParts = Object.entries(enabledAreas)
          .filter(([_, enabled]) => enabled)
          .map(([id]) => id)

        const response = await api.post<ProcessedImage>('/process', {
          image: base64Data,
          effect,
          enabled_parts: enabledParts,
          strength,
          sticker_category: effect === 'sticker' ? stickerCategory : undefined
        })

        setProcessedImage(response.data.processed_image)
      }
    } catch (err: any) {
      Logger.error('Error processing image:', err)
      setError(err.response?.data?.error || 'An error occurred while processing the image')
    } finally {
      setIsProcessing(false)
    }
  }

  const toggleArea = (areaId: string) => {
    setEnabledAreas(prev => ({
      ...prev,
      [areaId]: !prev[areaId]
    }))
  }

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-gray-100">
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
            Image Censor
          </h1>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="md:col-span-2 space-y-6">
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200 ${
                isDragActive
                  ? 'border-purple-500 bg-purple-500/10 shadow-lg shadow-purple-500/20'
                  : 'border-gray-800 hover:border-purple-500/50 hover:shadow-lg hover:shadow-purple-500/10'
              }`}
            >
              <input {...getInputProps()} />
              {preview ? (
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-96 mx-auto rounded-lg shadow-2xl"
                />
              ) : (
                <div className="flex flex-col items-center">
                  <div className="p-4 rounded-full mb-4 bg-gray-800">
                    <svg
                      className="h-8 w-8 text-purple-400"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                      />
                    </svg>
                  </div>
                  <p className="text-gray-400">
                    Drag & drop an image here, or click to select one
                  </p>
                </div>
              )}
            </div>

            {processedImage && (
              <div className="p-6 rounded-lg transition-all duration-200 bg-gray-800/50 shadow-lg shadow-purple-500/10">
                <h2 className="text-xl font-semibold mb-4 bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
                  Processed Image
                </h2>
                <img
                  src={`data:image/png;base64,${processedImage}`}
                  alt="Processed"
                  className="max-h-96 mx-auto rounded-lg shadow-2xl"
                />
              </div>
            )}

            {error && (
              <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-lg">
                {error}
              </div>
            )}

            {effect !== 'blackbox' && effect !== 'ruin' && effect !== 'sticker' && (
              <div className="p-6 rounded-lg bg-gray-800/50 shadow-lg shadow-purple-500/10">
                <h2 className="text-xl font-semibold mb-4 bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
                  Effect Strength
                </h2>
                <p className="text-gray-400 text-sm mb-4">
                  {effect === 'blackbox' 
                    ? 'Strength setting does not affect the black box effect'
                    : effect === 'ruin'
                    ? 'Strength setting does not affect the ruin effect as it uses predefined values'
                    : `Adjust the intensity of the ${effect} effect (1 = subtle, 10 = strong)`
                  }
                </p>
                <div className="flex items-center space-x-4">
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={strength}
                    onChange={(e) => setStrength(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    disabled={effect === 'blackbox' || effect === 'ruin'}
                  />
                  <span className="text-gray-300 min-w-[2rem] text-center">{strength}</span>
                </div>
              </div>
            )}
          </div>

          <div className="p-6 rounded-lg transition-all duration-200 bg-gray-800/50 backdrop-blur-sm shadow-lg shadow-purple-500/10">
            <h2 className="text-xl font-semibold mb-6 bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
              Detection Settings
            </h2>
            
            <div className="mb-6">
              <label className="block text-sm font-medium mb-2">Effect Type</label>
              <select
                value={effect}
                onChange={(e) => setEffect(e.target.value as Effect)}
                className="w-full p-2 rounded-lg transition-all duration-200 bg-gray-900 border-gray-700 text-gray-100 focus:border-purple-500 focus:ring-purple-500 border focus:ring-2 focus:ring-opacity-50"
              >
                <option value="pixelation">Pixelation</option>
                <option value="blur">Blur</option>
                <option value="blackbox">Black Box</option>
                <option value="ruin">Ruin</option>
                <option value="sticker">Stickers</option>
              </select>
            </div>

            {effect === 'sticker' && (
              <div className="mb-6">
                <label className="block text-sm font-medium mb-2">Sticker Category</label>
                <select
                  value={stickerCategory}
                  onChange={(e) => setStickerCategory(e.target.value)}
                  className="w-full p-2 rounded-lg transition-all duration-200 bg-gray-900 border-gray-700 text-gray-100 focus:border-purple-500 focus:ring-purple-500 border focus:ring-2 focus:ring-opacity-50"
                >
                  {STICKER_CATEGORIES.map((category) => (
                    <option key={category.id} value={category.id}>
                      {category.name}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <DetectionMenu
              enabled={enabledAreas}
              onToggle={toggleArea}
              darkMode={true}
            />

            {effectUsesStrength(effect) && (
              <div className="p-6 rounded-lg bg-gray-800/50 shadow-lg shadow-purple-500/10">
                <h2 className="text-xl font-semibold mb-4 bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
                  Effect Strength
                </h2>
                <p className="text-gray-400 text-sm mb-4">
                  Adjust the intensity of the {effect} effect (1 = subtle, 10 = strong)
                </p>
                <div className="flex items-center space-x-4">
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={strength}
                    onChange={(e) => setStrength(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                  <span className="text-gray-300 min-w-[2rem] text-center">{strength}</span>
                </div>
              </div>
            )}

            <button
              onClick={handleProcess}
              disabled={!file || isProcessing}
              className={`w-full mt-6 py-2 px-4 rounded-lg font-medium transition-all duration-200 ${
                !file || isProcessing
                  ? 'bg-gray-600 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:shadow-lg hover:shadow-purple-500/20'
              } text-white`}
            >
              {isProcessing ? (
                <div className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </div>
              ) : (
                'Process Image'
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
