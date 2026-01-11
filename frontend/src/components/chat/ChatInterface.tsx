'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, FileText, Paperclip, X } from 'lucide-react'
import { ChatMessage } from './ChatMessage'
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
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendStreamingMessage = async (message: string) => {
    setIsStreaming(true)
    abortControllerRef.current = new AbortController()

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

      // Create initial empty assistant message
      const initialMessage: Message = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
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
                case 'start':
                  console.log('Stream started:', eventData.conversation_id)
                  break

                case 'token':
                  streamedContent += eventData.content || ''
                  // Update message with new content
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
                  // Update message with sources
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId
                        ? { ...msg, sources }
                        : msg
                    )
                  )
                  break

                case 'done':
                  console.log('Stream completed:', {
                    tokens: eventData.tokens_used,
                    confidence: eventData.confidence,
                  })
                  break

                case 'error':
                  console.error('Stream error:', eventData.error)
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

  const sendMessageMutation = useMutation({
    mutationFn: async (message: string) => {
      const response = await axios.post('/api/v1/chat', {
        message,
        use_rag: true,
        temperature: 0.7,
      })
      return response.data
    },
    onSuccess: (data) => {
      const assistantMessage: Message = {
        id: Date.now().toString() + '-assistant',
        role: 'assistant',
        content: data.message,
        timestamp: new Date(data.timestamp),
        sources: data.sources,
      }
      setMessages((prev) => [...prev, assistantMessage])
    },
    onError: (error) => {
      console.error('Chat error:', error)
      const errorMessage: Message = {
        id: Date.now().toString() + '-error',
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    },
  })

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    setUploadedFiles((prev) => [...prev, ...files])
  }

  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if ((!input.trim() && uploadedFiles.length === 0) || isStreaming) return

    let messageContent = input

    // If files are uploaded, mention them in the message
    if (uploadedFiles.length > 0) {
      const fileNames = uploadedFiles.map(f => f.name).join(', ')
      messageContent = `${input}\n\n[Attached files: ${fileNames}]`
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: messageContent,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    const messageToSend = input
    setInput('')
    setUploadedFiles([])

    // Use streaming by default
    await sendStreamingMessage(messageToSend)
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-white dark:bg-background">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-4 py-6 bg-gray-50 dark:bg-background">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <FileText className="w-16 h-16 text-gray-400 dark:text-text-secondary mx-auto mb-4 opacity-50" />
              <p className="text-gray-600 dark:text-text-secondary">
                Start a conversation by typing your question below
              </p>
            </div>
          )}

          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}

          {(sendMessageMutation.isPending || isStreaming) && messages[messages.length - 1]?.role === 'user' && (
            <div className="flex justify-start">
              <div className="flex items-center gap-2 px-4 py-3 bg-gray-100 dark:bg-background-secondary rounded-lg">
                <Loader2 className="w-4 h-4 animate-spin text-brand-cyan" />
                <span className="text-sm text-gray-600 dark:text-text-secondary">
                  {isStreaming ? 'Streaming response...' : 'Thinking...'}
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
          {/* File Upload Preview */}
          {uploadedFiles.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {uploadedFiles.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg text-sm"
                >
                  <FileText className="w-4 h-4 text-brand-cyan" />
                  <span className="text-gray-900 dark:text-text-primary">{file.name}</span>
                  <button
                    type="button"
                    onClick={() => removeFile(index)}
                    className="text-gray-600 dark:text-text-secondary hover:text-red-400 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="relative flex items-end gap-2">
            {/* File Upload Button */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.doc,.docx,.csv,.json"
              onChange={handleFileChange}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="p-3 bg-gray-50 dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg hover:border-brand-cyan transition-all duration-200"
              title="Attach files"
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
