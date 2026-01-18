'use client'

import { useState } from 'react'
import {
  Image as ImageIcon,
  Video,
  Music,
  Box,
  Wand2,
  Upload,
  Download,
  Sparkles
} from 'lucide-react'

type MediaType = 'image' | 'video' | 'audio' | '3d' | 'deepfake'
type ActionType = 'generate' | 'edit' | 'transform'

interface GenerationRequest {
  type: MediaType
  action: ActionType
  prompt?: string
  files?: File[]
  options?: Record<string, any>
}

export function MultimodalInterface() {
  const [selectedType, setSelectedType] = useState<MediaType>('image')
  const [selectedAction, setSelectedAction] = useState<ActionType>('generate')
  const [prompt, setPrompt] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [generatedMedia, setGeneratedMedia] = useState<any[]>([])
  const [isGenerating, setIsGenerating] = useState(false)

  const mediaTypes = [
    { id: 'image' as MediaType, label: 'Images', icon: ImageIcon, color: 'text-blue-500' },
    { id: 'video' as MediaType, label: 'Videos', icon: Video, color: 'text-purple-500' },
    { id: 'audio' as MediaType, label: 'Audio', icon: Music, color: 'text-green-500' },
    { id: '3d' as MediaType, label: '3D Models', icon: Box, color: 'text-orange-500' },
    { id: 'deepfake' as MediaType, label: 'Deepfake', icon: Wand2, color: 'text-pink-500' }
  ]

  const actions = [
    { id: 'generate' as ActionType, label: 'Generate', description: 'Create from scratch' },
    { id: 'edit' as ActionType, label: 'Edit', description: 'Modify existing' },
    { id: 'transform' as ActionType, label: 'Transform', description: 'Advanced effects' }
  ]

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    setUploadedFiles(prev => [...prev, ...files])
  }

  const handleGenerate = async () => {
    if (!prompt && uploadedFiles.length === 0) {
      alert('Please enter a prompt or upload files')
      return
    }

    setIsGenerating(true)

    try {
      let endpoint = ''
      let formData = new FormData()

      // Determine endpoint based on type and action
      if (selectedType === 'image') {
        if (selectedAction === 'generate') {
          endpoint = '/api/v1/multimodal/generate/image'
          // Send JSON for image generation
          const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              prompt,
              style: 'realistic',
              width: 1024,
              height: 1024,
              num_images: 1
            })
          })
          const data = await response.json()
          setGeneratedMedia(prev => [...prev, ...data.images])
        } else if (selectedAction === 'edit') {
          endpoint = '/api/v1/multimodal/edit/image/upscale'
          formData.append('image', uploadedFiles[0])
          formData.append('scale_factor', '4')

          const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
          })
          const data = await response.json()
          setGeneratedMedia(prev => [...prev, data])
        }
      } else if (selectedType === 'video') {
        endpoint = '/api/v1/multimodal/generate/video'
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt,
            duration_seconds: 5,
            fps: 24,
            width: 576,
            height: 320
          })
        })
        const data = await response.json()
        setGeneratedMedia(prev => [...prev, data])
      } else if (selectedType === 'audio') {
        endpoint = '/api/v1/multimodal/generate/audio/tts'
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: prompt,
            language: 'en'
          })
        })
        const data = await response.json()
        setGeneratedMedia(prev => [...prev, data])
      } else if (selectedType === '3d') {
        endpoint = '/api/v1/multimodal/generate/3d'
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt,
            output_format: 'glb'
          })
        })
        const data = await response.json()
        setGeneratedMedia(prev => [...prev, data])
      } else if (selectedType === 'deepfake') {
        if (uploadedFiles.length < 2) {
          alert('Please upload source and target images for deepfake')
          return
        }
        endpoint = '/api/v1/multimodal/deepfake/face_swap'
        formData.append('source_image', uploadedFiles[0])
        formData.append('target_image', uploadedFiles[1])
        formData.append('face_enhancement', 'true')
        formData.append('super_resolution', 'true')

        const response = await fetch(endpoint, {
          method: 'POST',
          body: formData
        })
        const data = await response.json()
        setGeneratedMedia(prev => [...prev, data])
      }
    } catch (error) {
      console.error('Generation failed:', error)
      alert('Generation failed. Please try again.')
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="h-full flex flex-col bg-white dark:bg-background">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-white/10 p-4">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="w-6 h-6 text-brand-cyan" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-text-primary">
            Multimodal AI Studio
          </h2>
        </div>

        {/* Media Type Selection */}
        <div className="grid grid-cols-5 gap-2">
          {mediaTypes.map(type => (
            <button
              key={type.id}
              onClick={() => setSelectedType(type.id)}
              className={`p-3 rounded-lg border-2 transition-all ${
                selectedType === type.id
                  ? 'border-brand-cyan bg-brand-cyan/10'
                  : 'border-gray-200 dark:border-white/10 hover:border-brand-cyan/50'
              }`}
            >
              <type.icon className={`w-6 h-6 mx-auto mb-1 ${type.color}`} />
              <span className="text-xs font-medium text-gray-900 dark:text-text-primary">
                {type.label}
              </span>
            </button>
          ))}
        </div>

        {/* Action Selection */}
        <div className="grid grid-cols-3 gap-2 mt-4">
          {actions.map(action => (
            <button
              key={action.id}
              onClick={() => setSelectedAction(action.id)}
              className={`p-2 rounded-lg border transition-all ${
                selectedAction === action.id
                  ? 'border-brand-cyan bg-brand-cyan/10'
                  : 'border-gray-200 dark:border-white/10 hover:border-brand-cyan/50'
              }`}
            >
              <div className="text-sm font-medium text-gray-900 dark:text-text-primary">
                {action.label}
              </div>
              <div className="text-xs text-gray-500 dark:text-text-secondary">
                {action.description}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Input Area */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {/* Prompt Input */}
          <div>
            <label className="block text-sm font-medium text-gray-900 dark:text-text-primary mb-2">
              Prompt / Description
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={`Describe what you want to ${selectedAction}...`}
              rows={4}
              className="w-full px-4 py-3 bg-gray-50 dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg text-gray-900 dark:text-text-primary placeholder:text-gray-500 dark:placeholder:text-text-secondary focus:outline-none focus:border-brand-cyan focus:ring-2 focus:ring-brand-cyan/20"
            />
          </div>

          {/* File Upload */}
          {selectedAction !== 'generate' || selectedType === 'deepfake' ? (
            <div>
              <label className="block text-sm font-medium text-gray-900 dark:text-text-primary mb-2">
                Upload Files
              </label>
              <div className="border-2 border-dashed border-gray-300 dark:border-white/20 rounded-lg p-6 text-center">
                <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <input
                  type="file"
                  multiple
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                  accept={selectedType === 'image' ? 'image/*' : selectedType === 'video' ? 'video/*' : selectedType === 'audio' ? 'audio/*' : '*'}
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer text-brand-cyan hover:text-brand-cyan/80 font-medium"
                >
                  Click to upload or drag and drop
                </label>
                <p className="text-sm text-gray-500 dark:text-text-secondary mt-1">
                  {uploadedFiles.length} file(s) selected
                </p>
              </div>
            </div>
          ) : null}

          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={isGenerating}
            className="w-full py-3 px-6 bg-brand-cyan text-white rounded-lg hover:bg-brand-cyan/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2"
          >
            {isGenerating ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                {selectedAction.charAt(0).toUpperCase() + selectedAction.slice(1)} {selectedType}
              </>
            )}
          </button>

          {/* Generated Media Gallery */}
          {generatedMedia.length > 0 && (
            <div className="mt-8">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-text-primary mb-4">
                Generated Media ({generatedMedia.length})
              </h3>
              <div className="grid grid-cols-2 gap-4">
                {generatedMedia.map((media, idx) => (
                  <div
                    key={idx}
                    className="border border-gray-200 dark:border-white/10 rounded-lg p-4 bg-gray-50 dark:bg-background-secondary"
                  >
                    {/* Display based on type */}
                    {selectedType === 'image' && media.image && (
                      <img
                        src={`data:image/png;base64,${media.image}`}
                        alt="Generated"
                        className="w-full rounded"
                      />
                    )}
                    {selectedType === 'video' && media.video && (
                      <video
                        src={`data:video/mp4;base64,${media.video}`}
                        controls
                        className="w-full rounded"
                      />
                    )}
                    {selectedType === 'audio' && media.audio && (
                      <audio
                        src={`data:audio/wav;base64,${media.audio}`}
                        controls
                        className="w-full"
                      />
                    )}
                    <button className="mt-2 w-full py-2 px-4 bg-brand-cyan text-white rounded hover:bg-brand-cyan/90 transition-colors flex items-center justify-center gap-2">
                      <Download className="w-4 h-4" />
                      Download
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
