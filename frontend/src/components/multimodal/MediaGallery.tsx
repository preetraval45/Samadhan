'use client'

import { useState, useEffect } from 'react'
import { Download, Trash2, Eye, Share2, Filter, Grid, List } from 'lucide-react'

interface MediaItem {
  id: string
  type: 'image' | 'video' | 'audio' | '3d'
  url: string
  thumbnail?: string
  title: string
  createdAt: Date
  metadata?: Record<string, any>
}

export function MediaGallery() {
  const [mediaItems, setMediaItems] = useState<MediaItem[]>([])
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [filterType, setFilterType] = useState<string>('all')
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set())

  useEffect(() => {
    // Load media from localStorage
    const saved = localStorage.getItem('generatedMedia')
    if (saved) {
      const parsed = JSON.parse(saved)
      setMediaItems(parsed.map((item: any) => ({
        ...item,
        createdAt: new Date(item.createdAt)
      })))
    }
  }, [])

  const filteredItems = filterType === 'all'
    ? mediaItems
    : mediaItems.filter(item => item.type === filterType)

  const downloadItem = (item: MediaItem) => {
    // Create download link
    const link = document.createElement('a')
    link.href = item.url
    link.download = `${item.title}.${item.type === 'image' ? 'png' : item.type === 'video' ? 'mp4' : item.type === 'audio' ? 'wav' : 'glb'}`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const deleteItem = (id: string) => {
    const updated = mediaItems.filter(item => item.id !== id)
    setMediaItems(updated)
    localStorage.setItem('generatedMedia', JSON.stringify(updated))
  }

  const toggleSelect = (id: string) => {
    const newSelected = new Set(selectedItems)
    if (newSelected.has(id)) {
      newSelected.delete(id)
    } else {
      newSelected.add(id)
    }
    setSelectedItems(newSelected)
  }

  const deleteSelected = () => {
    const updated = mediaItems.filter(item => !selectedItems.has(item.id))
    setMediaItems(updated)
    setSelectedItems(new Set())
    localStorage.setItem('generatedMedia', JSON.stringify(updated))
  }

  return (
    <div className="h-full flex flex-col bg-white dark:bg-background">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-white/10 p-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-text-primary">
            Media Gallery
          </h2>
          <div className="flex items-center gap-2">
            {/* View Mode Toggle */}
            <div className="flex bg-gray-100 dark:bg-background-secondary rounded-lg p-1">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded ${viewMode === 'grid' ? 'bg-white dark:bg-background-tertiary shadow' : ''}`}
              >
                <Grid className="w-4 h-4 text-gray-600 dark:text-text-secondary" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded ${viewMode === 'list' ? 'bg-white dark:bg-background-tertiary shadow' : ''}`}
              >
                <List className="w-4 h-4 text-gray-600 dark:text-text-secondary" />
              </button>
            </div>

            {/* Delete Selected */}
            {selectedItems.size > 0 && (
              <button
                onClick={deleteSelected}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors flex items-center gap-2"
              >
                <Trash2 className="w-4 h-4" />
                Delete ({selectedItems.size})
              </button>
            )}
          </div>
        </div>

        {/* Filters */}
        <div className="flex gap-2">
          {['all', 'image', 'video', 'audio', '3d'].map(type => (
            <button
              key={type}
              onClick={() => setFilterType(type)}
              className={`px-4 py-2 rounded-lg transition-all ${
                filterType === type
                  ? 'bg-brand-cyan text-white'
                  : 'bg-gray-100 dark:bg-background-secondary text-gray-600 dark:text-text-secondary hover:bg-gray-200 dark:hover:bg-background-tertiary'
              }`}
            >
              {type.charAt(0).toUpperCase() + type.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Gallery */}
      <div className="flex-1 overflow-y-auto p-4">
        {filteredItems.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-500 dark:text-text-secondary">
            <Filter className="w-16 h-16 mb-4 opacity-50" />
            <p className="text-lg">No media items yet</p>
            <p className="text-sm">Generate some content to see it here</p>
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {filteredItems.map(item => (
              <div
                key={item.id}
                className={`border-2 rounded-lg overflow-hidden transition-all cursor-pointer ${
                  selectedItems.has(item.id)
                    ? 'border-brand-cyan shadow-lg'
                    : 'border-gray-200 dark:border-white/10 hover:border-brand-cyan/50'
                }`}
                onClick={() => toggleSelect(item.id)}
              >
                {/* Thumbnail */}
                <div className="aspect-square bg-gray-100 dark:bg-background-secondary relative">
                  {item.type === 'image' ? (
                    <img
                      src={item.url}
                      alt={item.title}
                      className="w-full h-full object-cover"
                    />
                  ) : item.type === 'video' ? (
                    <video
                      src={item.url}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-400">
                      {item.type.toUpperCase()}
                    </div>
                  )}
                  {selectedItems.has(item.id) && (
                    <div className="absolute top-2 right-2 w-6 h-6 bg-brand-cyan rounded-full flex items-center justify-center">
                      <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" />
                      </svg>
                    </div>
                  )}
                </div>

                {/* Info */}
                <div className="p-3 bg-white dark:bg-background-secondary">
                  <p className="text-sm font-medium text-gray-900 dark:text-text-primary truncate">
                    {item.title}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-text-secondary">
                    {item.createdAt.toLocaleDateString()}
                  </p>

                  {/* Actions */}
                  <div className="flex gap-1 mt-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        downloadItem(item)
                      }}
                      className="flex-1 p-1.5 bg-gray-100 dark:bg-background-tertiary rounded hover:bg-brand-cyan hover:text-white transition-colors"
                      title="Download"
                    >
                      <Download className="w-4 h-4 mx-auto" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteItem(item.id)
                      }}
                      className="flex-1 p-1.5 bg-gray-100 dark:bg-background-tertiary rounded hover:bg-red-500 hover:text-white transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4 mx-auto" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {filteredItems.map(item => (
              <div
                key={item.id}
                className={`flex items-center gap-4 p-4 border-2 rounded-lg transition-all cursor-pointer ${
                  selectedItems.has(item.id)
                    ? 'border-brand-cyan bg-brand-cyan/5'
                    : 'border-gray-200 dark:border-white/10 hover:border-brand-cyan/50'
                }`}
                onClick={() => toggleSelect(item.id)}
              >
                {/* Thumbnail */}
                <div className="w-20 h-20 bg-gray-100 dark:bg-background-secondary rounded overflow-hidden">
                  {item.type === 'image' && (
                    <img src={item.url} alt={item.title} className="w-full h-full object-cover" />
                  )}
                </div>

                {/* Info */}
                <div className="flex-1">
                  <h3 className="font-medium text-gray-900 dark:text-text-primary">{item.title}</h3>
                  <p className="text-sm text-gray-500 dark:text-text-secondary">
                    {item.type.toUpperCase()} • {item.createdAt.toLocaleDateString()}
                  </p>
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      downloadItem(item)
                    }}
                    className="p-2 bg-gray-100 dark:bg-background-secondary rounded-lg hover:bg-brand-cyan hover:text-white transition-colors"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      deleteItem(item.id)
                    }}
                    className="p-2 bg-gray-100 dark:bg-background-secondary rounded-lg hover:bg-red-500 hover:text-white transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
