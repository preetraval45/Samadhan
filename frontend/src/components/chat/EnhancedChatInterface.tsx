'use client'

import { useState, useRef, useEffect } from 'react'
import {
  Send,
  Loader2,
  FileText,
  Paperclip,
  X,
  RefreshCw,
  Edit3,
  GitBranch,
  Copy,
  Check,
} from 'lucide-react'
import { ChatMessage } from './ChatMessage'
import { EnhancedMessage } from './EnhancedMessage'
import { SmartAttachments } from './SmartAttachments'
import { useMutation } from '@tanstack/react-query'
import axios from 'axios'

interface StreamEvent {
  type: 'start' | 'token' | 'sources' | 'done' | 'error'
  content?: string
  conversation_id?: string
  model?: string
  sources?: Array<{
    document: string
    page: number
    relevance_score: number
    snippet: string
  }>
  tokens_used?: number
  confidence?: number
  timestamp?: string
  error?: string
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  sources?: Array<{
    document: string
    page: number
    relevance_score: number
    snippet: string
  }>
  isEdited?: boolean
  branches?: Message[]
  parentId?: string
  model?: string
  attachments?: Array<{
    name: string
    type: string
    size: number
    preview?: string
  }>
}

interface EnhancedChatInterfaceProps {
  tabId: string
  onTitleUpdate?: (title: string) => void
}

export function EnhancedChatInterface({
  tabId,
  onTitleUpdate,
}: EnhancedChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null)
  const [editedContent, setEditedContent] = useState('')
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<string>('auto')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Auto-update tab title from first message
  useEffect(() => {
    if (messages.length > 0 && onTitleUpdate) {
      const firstUserMessage = messages.find((m) => m.role === 'user')
      if (firstUserMessage) {
        const title = firstUserMessage.content.slice(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '')
        onTitleUpdate(title)
      }
    }
  }, [messages, onTitleUpdate])

  // Model auto-selection based on query type
  const detectQueryType = (query: string): string => {
    const lower = query.toLowerCase()

    if (
      lower.includes('code') ||
      lower.includes('function') ||
      lower.includes('debug') ||
      lower.includes('implement')
    ) {
      return 'code'
    }

    if (
      lower.includes('image') ||
      lower.includes('picture') ||
      lower.includes('generate') ||
      lower.includes('create')
    ) {
      return 'image'
    }

    if (
      lower.includes('calculate') ||
      lower.includes('solve') ||
      lower.includes('math') ||
      lower.includes('equation')
    ) {
      return 'reasoning'
    }

    if (
      lower.includes('story') ||
      lower.includes('creative') ||
      lower.includes('write') ||
      lower.includes('poem')
    ) {
      return 'creative'
    }

    return 'general'
  }

  const sendStreamingMessage = async (message: string) => {
    setIsStreaming(true)
    abortControllerRef.current = new AbortController()

    const queryType = detectQueryType(message)
    const modelToUse = selectedModel === 'auto' ? queryType : selectedModel

    try {
      const response = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          use_rag: true,
          temperature: 0.7,
          model_type: modelToUse,
          conversation_id: tabId,
        }),
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        throw new Error('Streaming failed')
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      let streamedContent = ''
      let assistantMessageId = Date.now().toString() + '-assistant'
      let sources: Message['sources'] = []

      const initialMessage: Message = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        model: modelToUse,
      }
      setMessages((prev) => [...prev, initialMessage])

      if (!reader) return

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const eventData: StreamEvent = JSON.parse(line.slice(6))

              switch (eventData.type) {
                case 'token':
                  streamedContent += eventData.content || ''
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId
                        ? { ...msg, content: streamedContent }
                        : msg
                    )
                  )
                  break

                case 'sources':
                  sources = eventData.sources || []
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId ? { ...msg, sources } : msg
                    )
                  )
                  break

                case 'error':
                  streamedContent = 'Sorry, I encountered an error. Please try again.'
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId
                        ? { ...msg, content: streamedContent }
                        : msg
                    )
                  )
                  break
              }
            } catch (error) {
              console.error('Error parsing stream event:', error)
            }
          }
        }
      }
    } catch (error: any) {
      if (error.name !== 'AbortError') {
        console.error('Streaming error:', error)
        const errorMessage: Message = {
          id: Date.now().toString() + '-error',
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, errorMessage])
      }
    } finally {
      setIsStreaming(false)
      abortControllerRef.current = null
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if ((!input.trim() && uploadedFiles.length === 0) || isStreaming) return

    let messageContent = input

    if (uploadedFiles.length > 0) {
      const fileNames = uploadedFiles.map((f) => f.name).join(', ')
      messageContent = `${input}\n\n[Attached files: ${fileNames}]`
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: messageContent,
      timestamp: new Date(),
      attachments: uploadedFiles.map((f) => ({
        name: f.name,
        type: f.type,
        size: f.size,
      })),
    }

    setMessages((prev) => [...prev, userMessage])
    const messageToSend = input
    setInput('')
    setUploadedFiles([])

    await sendStreamingMessage(messageToSend)
  }

  const handleRegenerateResponse = async (messageId: string) => {
    const messageIndex = messages.findIndex((m) => m.id === messageId)
    if (messageIndex === -1) return

    const previousUserMessage = messages
      .slice(0, messageIndex)
      .reverse()
      .find((m) => m.role === 'user')

    if (previousUserMessage) {
      // Remove old assistant message
      setMessages((prev) => prev.filter((m) => m.id !== messageId))

      // Regenerate
      await sendStreamingMessage(previousUserMessage.content)
    }
  }

  const handleEditMessage = (messageId: string) => {
    const message = messages.find((m) => m.id === messageId)
    if (message) {
      setEditingMessageId(messageId)
      setEditedContent(message.content)
    }
  }

  const saveEditedMessage = async () => {
    if (!editingMessageId || !editedContent.trim()) return

    setMessages((prev) =>
      prev.map((m) =>
        m.id === editingMessageId
          ? { ...m, content: editedContent, isEdited: true }
          : m
      )
    )

    // Regenerate responses after this message
    const messageIndex = messages.findIndex((m) => m.id === editingMessageId)
    setMessages((prev) => prev.slice(0, messageIndex + 1))

    setEditingMessageId(null)
    setEditedContent('')

    // Send new version
    await sendStreamingMessage(editedContent)
  }

  const handleCopyMessage = (messageId: string, content: string) => {
    navigator.clipboard.writeText(content)
    setCopiedMessageId(messageId)
    setTimeout(() => setCopiedMessageId(null), 2000)
  }

  const handleBranchConversation = (messageId: string) => {
    // Create a new branch from this message
    const messageIndex = messages.findIndex((m) => m.id === messageId)
    if (messageIndex === -1) return

    const branchedMessages = messages.slice(0, messageIndex + 1)
    // In a real implementation, this would create a new tab or save the branch
    console.log('Branch created at message:', messageId, branchedMessages)
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-white dark:bg-background">
      {/* Model Selector Bar */}
      <div className="px-4 py-2 bg-gray-50 dark:bg-background-secondary border-b border-gray-200 dark:border-white/10">
        <div className="max-w-4xl mx-auto flex items-center gap-4">
          <label className="text-sm text-gray-600 dark:text-text-secondary">
            Model:
          </label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="text-sm px-3 py-1.5 bg-white dark:bg-background border border-gray-200 dark:border-white/10 rounded-lg text-gray-900 dark:text-text-primary focus:outline-none focus:border-brand-cyan"
          >
            <option value="auto">Auto-Select (Recommended)</option>
            <option value="code">Code-Optimized</option>
            <option value="image">Vision</option>
            <option value="reasoning">Reasoning</option>
            <option value="creative">Creative Writing</option>
            <option value="general">General</option>
          </select>
          {selectedModel === 'auto' && (
            <span className="text-xs text-gray-500 dark:text-text-secondary">
              Automatically selects best model for your query
            </span>
          )}
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-4 py-6 bg-gray-50 dark:bg-background">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <FileText className="w-16 h-16 text-gray-400 dark:text-text-secondary mx-auto mb-4 opacity-50" />
              <p className="text-gray-600 dark:text-text-secondary">
                Start a conversation by typing your question below
              </p>
              <p className="text-sm text-gray-500 dark:text-text-secondary mt-2">
                The model will automatically adapt to your query type
              </p>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className="group relative">
              {editingMessageId === message.id ? (
                <div className="space-y-2">
                  <textarea
                    value={editedContent}
                    onChange={(e) => setEditedContent(e.target.value)}
                    className="w-full px-4 py-3 bg-white dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg text-gray-900 dark:text-text-primary resize-none"
                    rows={4}
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={saveEditedMessage}
                      className="px-3 py-1.5 bg-brand-cyan text-white rounded-lg text-sm hover:bg-brand-cyan/90"
                    >
                      Save & Submit
                    </button>
                    <button
                      onClick={() => setEditingMessageId(null)}
                      className="px-3 py-1.5 bg-gray-200 dark:bg-background-secondary text-gray-900 dark:text-text-primary rounded-lg text-sm hover:bg-gray-300 dark:hover:bg-background"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <EnhancedMessage message={message} />

                  {/* Message Actions */}
                  <div className="absolute -right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1 bg-white dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg shadow-lg p-1">
                    <button
                      onClick={() =>
                        handleCopyMessage(message.id, message.content)
                      }
                      className="p-1.5 hover:bg-gray-100 dark:hover:bg-background rounded"
                      title="Copy"
                    >
                      {copiedMessageId === message.id ? (
                        <Check className="w-4 h-4 text-green-500" />
                      ) : (
                        <Copy className="w-4 h-4 text-gray-600 dark:text-text-secondary" />
                      )}
                    </button>

                    {message.role === 'user' && (
                      <button
                        onClick={() => handleEditMessage(message.id)}
                        className="p-1.5 hover:bg-gray-100 dark:hover:bg-background rounded"
                        title="Edit"
                      >
                        <Edit3 className="w-4 h-4 text-gray-600 dark:text-text-secondary" />
                      </button>
                    )}

                    {message.role === 'assistant' && (
                      <button
                        onClick={() => handleRegenerateResponse(message.id)}
                        className="p-1.5 hover:bg-gray-100 dark:hover:bg-background rounded"
                        title="Regenerate"
                      >
                        <RefreshCw className="w-4 h-4 text-gray-600 dark:text-text-secondary" />
                      </button>
                    )}

                    <button
                      onClick={() => handleBranchConversation(message.id)}
                      className="p-1.5 hover:bg-gray-100 dark:hover:bg-background rounded"
                      title="Branch conversation"
                    >
                      <GitBranch className="w-4 h-4 text-gray-600 dark:text-text-secondary" />
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}

          {isStreaming && messages[messages.length - 1]?.role === 'user' && (
            <div className="flex justify-start">
              <div className="flex items-center gap-2 px-4 py-3 bg-gray-100 dark:bg-background-secondary rounded-lg">
                <Loader2 className="w-4 h-4 animate-spin text-brand-cyan" />
                <span className="text-sm text-gray-600 dark:text-text-secondary">
                  Streaming response...
                </span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 dark:border-white/10 bg-white dark:bg-background-secondary/50 backdrop-blur-lg p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          {/* Smart Attachments Preview */}
          <SmartAttachments
            files={uploadedFiles}
            onRemove={(index) =>
              setUploadedFiles((prev) => prev.filter((_, i) => i !== index))
            }
          />

          <div className="relative flex items-end gap-2">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="*/*"
              onChange={(e) =>
                setUploadedFiles((prev) => [
                  ...prev,
                  ...Array.from(e.target.files || []),
                ])
              }
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="p-3 bg-gray-50 dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg hover:border-brand-cyan transition-all duration-200"
              title="Attach files (any type)"
            >
              <Paperclip className="w-5 h-5 text-gray-600 dark:text-text-secondary" />
            </button>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit(e)
                }
              }}
              placeholder="Ask anything... (Shift + Enter for new line)"
              className="flex-1 min-h-[52px] max-h-32 px-4 py-3 bg-gray-50 dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg text-gray-900 dark:text-text-primary placeholder:text-gray-500 dark:placeholder:text-text-secondary focus:outline-none focus:border-brand-cyan focus:ring-2 focus:ring-brand-cyan/20 transition-all duration-200 resize-none"
              rows={1}
            />
            <button
              type="submit"
              disabled={(!input.trim() && uploadedFiles.length === 0) || isStreaming}
              className="p-3 bg-gradient-brand rounded-lg hover:shadow-lg hover:shadow-brand-blue/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none"
            >
              {isStreaming ? (
                <Loader2 className="w-5 h-5 text-white animate-spin" />
              ) : (
                <Send className="w-5 h-5 text-white" />
              )}
            </button>
          </div>
          <p className="text-xs text-text-secondary mt-2 text-center">
            Samādhān may make mistakes. Please verify important information.
          </p>
        </form>
      </div>
    </div>
  )
}
