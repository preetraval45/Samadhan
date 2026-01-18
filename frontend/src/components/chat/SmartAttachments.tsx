'use client'

import { useState, useEffect } from 'react'
import {
  FileText,
  Image as ImageIcon,
  Video,
  Music,
  File,
  X,
  Eye,
  Loader2,
} from 'lucide-react'
import { useMutation } from '@tanstack/react-query'
import axios from 'axios'

interface SmartAttachmentsProps {
  files: File[]
  onRemove: (index: number) => void
}

interface FileAnalysis {
  type: string
  preview?: string
  text?: string
  transcription?: string
  metadata?: Record<string, any>
  isProcessing?: boolean
}

export function SmartAttachments({ files, onRemove }: SmartAttachmentsProps) {
  const [fileAnalyses, setFileAnalyses] = useState<Map<string, FileAnalysis>>(
    new Map()
  )
  const [expandedFile, setExpandedFile] = useState<string | null>(null)

  // Analyze files as they're added
  useEffect(() => {
    files.forEach((file) => {
      const fileKey = `${file.name}-${file.size}`
      if (!fileAnalyses.has(fileKey)) {
        analyzeFile(file, fileKey)
      }
    })
  }, [files])

  const analyzeFile = async (file: File, fileKey: string) => {
    const type = getFileType(file)

    // Set initial state
    setFileAnalyses((prev) =>
      new Map(prev).set(fileKey, { type, isProcessing: true })
    )

    try {
      // Generate preview for images
      if (type === 'image') {
        const preview = await generateImagePreview(file)
        setFileAnalyses((prev) => {
          const newMap = new Map(prev)
          const analysis = newMap.get(fileKey)!
          analysis.preview = preview
          analysis.isProcessing = false
          return newMap
        })

        // OCR for images
        const ocrText = await performOCR(file)
        if (ocrText) {
          setFileAnalyses((prev) => {
            const newMap = new Map(prev)
            const analysis = newMap.get(fileKey)!
            analysis.text = ocrText
            return newMap
          })
        }
      }

      // Extract text from PDFs
      else if (type === 'pdf') {
        const extractedText = await extractPDFText(file)
        setFileAnalyses((prev) => {
          const newMap = new Map(prev)
          const analysis = newMap.get(fileKey)!
          analysis.text = extractedText
          analysis.isProcessing = false
          return newMap
        })
      }

      // Transcribe audio
      else if (type === 'audio') {
        const transcription = await transcribeAudio(file)
        setFileAnalyses((prev) => {
          const newMap = new Map(prev)
          const analysis = newMap.get(fileKey)!
          analysis.transcription = transcription
          analysis.isProcessing = false
          return newMap
        })
      }

      // Analyze video
      else if (type === 'video') {
        const preview = await generateVideoThumbnail(file)
        setFileAnalyses((prev) => {
          const newMap = new Map(prev)
          const analysis = newMap.get(fileKey)!
          analysis.preview = preview
          analysis.isProcessing = false
          return newMap
        })
      }

      // Read code files
      else if (type === 'code') {
        const content = await file.text()
        setFileAnalyses((prev) => {
          const newMap = new Map(prev)
          const analysis = newMap.get(fileKey)!
          analysis.text = content
          analysis.isProcessing = false
          return newMap
        })
      }

      // Parse spreadsheets
      else if (type === 'spreadsheet') {
        const parsedData = await parseSpreadsheet(file)
        setFileAnalyses((prev) => {
          const newMap = new Map(prev)
          const analysis = newMap.get(fileKey)!
          analysis.metadata = parsedData
          analysis.isProcessing = false
          return newMap
        })
      }

      // Default: just mark as processed
      else {
        setFileAnalyses((prev) => {
          const newMap = new Map(prev)
          const analysis = newMap.get(fileKey)!
          analysis.isProcessing = false
          return newMap
        })
      }
    } catch (error) {
      console.error('Error analyzing file:', error)
      setFileAnalyses((prev) => {
        const newMap = new Map(prev)
        const analysis = newMap.get(fileKey)!
        analysis.isProcessing = false
        return newMap
      })
    }
  }

  const getFileType = (file: File): string => {
    const ext = file.name.split('.').pop()?.toLowerCase()

    if (file.type.startsWith('image/')) return 'image'
    if (file.type.startsWith('video/')) return 'video'
    if (file.type.startsWith('audio/')) return 'audio'
    if (file.type === 'application/pdf') return 'pdf'

    if (['js', 'ts', 'tsx', 'jsx', 'py', 'java', 'cpp', 'c', 'rs', 'go'].includes(ext || ''))
      return 'code'

    if (['csv', 'xlsx', 'xls'].includes(ext || '')) return 'spreadsheet'
    if (['doc', 'docx', 'txt', 'md'].includes(ext || '')) return 'document'

    return 'other'
  }

  const getFileIcon = (type: string) => {
    switch (type) {
      case 'image':
        return ImageIcon
      case 'video':
        return Video
      case 'audio':
        return Music
      default:
        return FileText
    }
  }

  const generateImagePreview = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const reader = new FileReader()
      reader.onload = (e) => resolve(e.target?.result as string)
      reader.readAsDataURL(file)
    })
  }

  const generateVideoThumbnail = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const video = document.createElement('video')
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')

      video.onloadeddata = () => {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        ctx?.drawImage(video, 0, 0)
        resolve(canvas.toDataURL())
      }

      video.src = URL.createObjectURL(file)
    })
  }

  const performOCR = async (file: File): Promise<string> => {
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/api/v1/multimodal/ocr', formData)
      return response.data.text || ''
    } catch (error) {
      console.error('OCR failed:', error)
      return ''
    }
  }

  const extractPDFText = async (file: File): Promise<string> => {
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/api/v1/documents/extract', formData)
      return response.data.text || ''
    } catch (error) {
      console.error('PDF extraction failed:', error)
      return ''
    }
  }

  const transcribeAudio = async (file: File): Promise<string> => {
    const formData = new FormData()
    formData.append('audio', file)

    try {
      const response = await axios.post('/api/v1/multimodal/transcribe', formData)
      return response.data.transcription || ''
    } catch (error) {
      console.error('Transcription failed:', error)
      return ''
    }
  }

  const parseSpreadsheet = async (file: File): Promise<Record<string, any>> => {
    // Simplified - in real app would use a library like xlsx
    return { rows: 0, columns: 0 }
  }

  if (files.length === 0) return null

  return (
    <div className="mb-3 space-y-2">
      {files.map((file, index) => {
        const fileKey = `${file.name}-${file.size}`
        const analysis = fileAnalyses.get(fileKey)
        const FileIcon = getFileIcon(analysis?.type || 'other')
        const isExpanded = expandedFile === fileKey

        return (
          <div
            key={fileKey}
            className="bg-white dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg overflow-hidden"
          >
            {/* File Header */}
            <div className="flex items-center gap-3 p-3">
              <div className="flex-shrink-0 w-10 h-10 bg-brand-cyan/10 rounded-lg flex items-center justify-center">
                {analysis?.isProcessing ? (
                  <Loader2 className="w-5 h-5 text-brand-cyan animate-spin" />
                ) : (
                  <FileIcon className="w-5 h-5 text-brand-cyan" />
                )}
              </div>

              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-gray-900 dark:text-text-primary truncate">
                  {file.name}
                </div>
                <div className="text-xs text-gray-500 dark:text-text-secondary flex items-center gap-2">
                  <span>{(file.size / 1024).toFixed(1)} KB</span>
                  {analysis?.isProcessing && (
                    <span className="text-brand-cyan">Processing...</span>
                  )}
                  {analysis?.text && (
                    <span className="text-green-500">Text extracted</span>
                  )}
                  {analysis?.transcription && (
                    <span className="text-green-500">Transcribed</span>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-1">
                {(analysis?.preview || analysis?.text || analysis?.transcription) && (
                  <button
                    onClick={() =>
                      setExpandedFile(isExpanded ? null : fileKey)
                    }
                    className="p-2 hover:bg-gray-100 dark:hover:bg-background rounded transition-colors"
                    title="Preview"
                  >
                    <Eye className="w-4 h-4 text-gray-600 dark:text-text-secondary" />
                  </button>
                )}

                <button
                  onClick={() => onRemove(index)}
                  className="p-2 hover:bg-red-100 dark:hover:bg-red-900/20 rounded transition-colors"
                  title="Remove"
                >
                  <X className="w-4 h-4 text-gray-600 dark:text-text-secondary hover:text-red-500" />
                </button>
              </div>
            </div>

            {/* Expanded Preview */}
            {isExpanded && analysis && (
              <div className="border-t border-gray-200 dark:border-white/10 p-3 bg-gray-50 dark:bg-background">
                {/* Image Preview */}
                {analysis.preview && analysis.type === 'image' && (
                  <img
                    src={analysis.preview}
                    alt={file.name}
                    className="max-w-full h-auto rounded-lg mb-2"
                  />
                )}

                {/* Video Thumbnail */}
                {analysis.preview && analysis.type === 'video' && (
                  <img
                    src={analysis.preview}
                    alt={file.name}
                    className="max-w-full h-auto rounded-lg mb-2"
                  />
                )}

                {/* Extracted Text */}
                {analysis.text && (
                  <div className="space-y-2">
                    <div className="text-xs font-medium text-gray-700 dark:text-text-secondary">
                      Extracted Text:
                    </div>
                    <div className="p-3 bg-white dark:bg-background-secondary rounded border border-gray-200 dark:border-white/10 text-sm text-gray-900 dark:text-text-primary max-h-40 overflow-y-auto">
                      {analysis.text.slice(0, 500)}
                      {analysis.text.length > 500 && '...'}
                    </div>
                  </div>
                )}

                {/* Audio Transcription */}
                {analysis.transcription && (
                  <div className="space-y-2">
                    <div className="text-xs font-medium text-gray-700 dark:text-text-secondary">
                      Transcription:
                    </div>
                    <div className="p-3 bg-white dark:bg-background-secondary rounded border border-gray-200 dark:border-white/10 text-sm text-gray-900 dark:text-text-primary max-h-40 overflow-y-auto">
                      {analysis.transcription}
                    </div>
                  </div>
                )}

                {/* Spreadsheet Metadata */}
                {analysis.metadata && (
                  <div className="text-xs text-gray-600 dark:text-text-secondary">
                    <pre>{JSON.stringify(analysis.metadata, null, 2)}</pre>
                  </div>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
