'use client'

import { useState } from 'react'
import {
  Search,
  X,
  Calendar,
  Filter,
  Download,
  FileText,
  Image as ImageIcon,
  BarChart3,
} from 'lucide-react'
import { useMutation, useQuery } from '@tanstack/react-query'
import axios from 'axios'

interface SearchFilters {
  dateRange?: {
    start: Date | null
    end: Date | null
  }
  modelUsed?: string[]
  hasAttachments?: boolean
  hasGenerated Media?: boolean
  keywords?: string[]
}

interface SearchResult {
  id: string
  conversationId: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  model?: string
  relevance: number
  highlights: string[]
}

interface AdvancedSearchProps {
  onClose: () => void
}

export function AdvancedSearch({ onClose }: AdvancedSearchProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [filters, setFilters] = useState<SearchFilters>({})
  const [showFilters, setShowFilters] = useState(false)
  const [selectedFormat, setSelectedFormat] = useState<'pdf' | 'markdown'>('pdf')

  // Semantic search query
  const searchMutation = useMutation({
    mutationFn: async (query: string) => {
      const response = await axios.post('/api/v1/search/conversations', {
        query,
        filters,
        semantic: true,
      })
      return response.data
    },
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchQuery.trim()) {
      searchMutation.mutate(searchQuery)
    }
  }

  const exportConversation = async (conversationId: string) => {
    try {
      const response = await axios.post(
        `/api/v1/conversations/${conversationId}/export`,
        { format: selectedFormat },
        { responseType: 'blob' }
      )

      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `conversation-${conversationId}.${selectedFormat}`)
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch (error) {
      console.error('Export failed:', error)
    }
  }

  const getAnalytics = useQuery({
    queryKey: ['conversation-analytics'],
    queryFn: async () => {
      const response = await axios.get('/api/v1/analytics/conversations')
      return response.data
    },
  })

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-background-secondary rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-white/10">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-text-primary">
              Advanced Search
            </h2>
            <p className="text-sm text-gray-600 dark:text-text-secondary mt-1">
              Semantic search across all conversations
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-background rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-600 dark:text-text-secondary" />
          </button>
        </div>

        {/* Search Bar */}
        <div className="p-6 border-b border-gray-200 dark:border-white/10">
          <form onSubmit={handleSearch} className="flex gap-2">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search conversations semantically..."
                className="w-full pl-10 pr-4 py-3 bg-gray-50 dark:bg-background border border-gray-200 dark:border-white/10 rounded-lg text-gray-900 dark:text-text-primary placeholder:text-gray-500 dark:placeholder:text-text-secondary focus:outline-none focus:border-brand-cyan focus:ring-2 focus:ring-brand-cyan/20"
              />
            </div>
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className={`p-3 rounded-lg border transition-colors ${
                showFilters
                  ? 'bg-brand-cyan text-white border-brand-cyan'
                  : 'bg-gray-50 dark:bg-background border-gray-200 dark:border-white/10 text-gray-600 dark:text-text-secondary hover:border-brand-cyan'
              }`}
            >
              <Filter className="w-5 h-5" />
            </button>
            <button
              type="submit"
              disabled={!searchQuery.trim() || searchMutation.isPending}
              className="px-6 py-3 bg-gradient-brand text-white rounded-lg hover:shadow-lg hover:shadow-brand-blue/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {searchMutation.isPending ? 'Searching...' : 'Search'}
            </button>
          </form>

          {/* Filters Panel */}
          {showFilters && (
            <div className="mt-4 p-4 bg-gray-50 dark:bg-background rounded-lg space-y-4">
              <div className="grid grid-cols-2 gap-4">
                {/* Date Range */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-text-secondary mb-2">
                    Date Range
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="date"
                      onChange={(e) =>
                        setFilters({
                          ...filters,
                          dateRange: {
                            ...filters.dateRange,
                            start: e.target.value ? new Date(e.target.value) : null,
                          },
                        })
                      }
                      className="flex-1 px-3 py-2 bg-white dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg text-sm"
                    />
                    <input
                      type="date"
                      onChange={(e) =>
                        setFilters({
                          ...filters,
                          dateRange: {
                            ...filters.dateRange,
                            end: e.target.value ? new Date(e.target.value) : null,
                          },
                        })
                      }
                      className="flex-1 px-3 py-2 bg-white dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg text-sm"
                    />
                  </div>
                </div>

                {/* Model Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-text-secondary mb-2">
                    Model Used
                  </label>
                  <select
                    multiple
                    onChange={(e) =>
                      setFilters({
                        ...filters,
                        modelUsed: Array.from(
                          e.target.selectedOptions,
                          (option) => option.value
                        ),
                      })
                    }
                    className="w-full px-3 py-2 bg-white dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg text-sm"
                  >
                    <option value="code">Code-Optimized</option>
                    <option value="image">Vision</option>
                    <option value="reasoning">Reasoning</option>
                    <option value="creative">Creative</option>
                    <option value="general">General</option>
                  </select>
                </div>
              </div>

              <div className="flex gap-4">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={filters.hasAttachments}
                    onChange={(e) =>
                      setFilters({ ...filters, hasAttachments: e.target.checked })
                    }
                    className="w-4 h-4 text-brand-cyan rounded border-gray-300 dark:border-white/10"
                  />
                  <span className="text-sm text-gray-700 dark:text-text-secondary">
                    Has Attachments
                  </span>
                </label>

                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={filters.hasGeneratedMedia}
                    onChange={(e) =>
                      setFilters({
                        ...filters,
                        hasGeneratedMedia: e.target.checked,
                      })
                    }
                    className="w-4 h-4 text-brand-cyan rounded border-gray-300 dark:border-white/10"
                  />
                  <span className="text-sm text-gray-700 dark:text-text-secondary">
                    Has Generated Media
                  </span>
                </label>
              </div>
            </div>
          )}
        </div>

        {/* Results */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {searchMutation.isPending && (
            <div className="text-center py-12">
              <div className="inline-block w-8 h-8 border-4 border-brand-cyan border-t-transparent rounded-full animate-spin" />
              <p className="mt-4 text-gray-600 dark:text-text-secondary">
                Searching conversations...
              </p>
            </div>
          )}

          {searchMutation.data?.results && (
            <>
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-600 dark:text-text-secondary">
                  Found {searchMutation.data.results.length} results
                </p>

                <div className="flex items-center gap-2">
                  <select
                    value={selectedFormat}
                    onChange={(e) =>
                      setSelectedFormat(e.target.value as 'pdf' | 'markdown')
                    }
                    className="text-sm px-3 py-1.5 bg-white dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg"
                  >
                    <option value="pdf">PDF</option>
                    <option value="markdown">Markdown</option>
                  </select>
                </div>
              </div>

              {searchMutation.data.results.map((result: SearchResult) => (
                <div
                  key={result.id}
                  className="p-4 bg-gray-50 dark:bg-background border border-gray-200 dark:border-white/10 rounded-lg hover:border-brand-cyan transition-colors"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs px-2 py-1 bg-brand-cyan/10 text-brand-cyan rounded-full">
                        {(result.relevance * 100).toFixed(0)}% relevant
                      </span>
                      {result.model && (
                        <span className="text-xs px-2 py-1 bg-gray-200 dark:bg-background-secondary text-gray-700 dark:text-text-secondary rounded-full">
                          {result.model}
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => exportConversation(result.conversationId)}
                      className="p-1.5 hover:bg-gray-200 dark:hover:bg-background-secondary rounded"
                      title="Export conversation"
                    >
                      <Download className="w-4 h-4 text-gray-600 dark:text-text-secondary" />
                    </button>
                  </div>

                  <div className="text-sm text-gray-900 dark:text-text-primary mb-2">
                    {result.highlights.map((highlight, i) => (
                      <span
                        key={i}
                        dangerouslySetInnerHTML={{ __html: highlight }}
                        className="[&>mark]:bg-yellow-200 dark:[&>mark]:bg-yellow-600 [&>mark]:text-gray-900"
                      />
                    ))}
                  </div>

                  <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-text-secondary">
                    <span>{new Date(result.timestamp).toLocaleDateString()}</span>
                    <span>{result.role === 'user' ? 'Your message' : 'AI response'}</span>
                  </div>
                </div>
              ))}
            </>
          )}

          {searchMutation.data?.results?.length === 0 && (
            <div className="text-center py-12">
              <FileText className="w-16 h-16 text-gray-400 dark:text-text-secondary mx-auto mb-4 opacity-50" />
              <p className="text-gray-600 dark:text-text-secondary">
                No results found. Try adjusting your search query or filters.
              </p>
            </div>
          )}
        </div>

        {/* Analytics Preview */}
        {getAnalytics.data && (
          <div className="p-6 border-t border-gray-200 dark:border-white/10 bg-gray-50 dark:bg-background">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="w-5 h-5 text-brand-cyan" />
              <h3 className="font-medium text-gray-900 dark:text-text-primary">
                Quick Stats
              </h3>
            </div>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-brand-cyan">
                  {getAnalytics.data.totalConversations}
                </div>
                <div className="text-xs text-gray-600 dark:text-text-secondary mt-1">
                  Conversations
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-500">
                  {getAnalytics.data.totalMessages}
                </div>
                <div className="text-xs text-gray-600 dark:text-text-secondary mt-1">
                  Messages
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-pink-500">
                  {getAnalytics.data.totalAttachments}
                </div>
                <div className="text-xs text-gray-600 dark:text-text-secondary mt-1">
                  Attachments
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-500">
                  {getAnalytics.data.totalGenerated}
                </div>
                <div className="text-xs text-gray-600 dark:text-text-secondary mt-1">
                  Generated Media
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
