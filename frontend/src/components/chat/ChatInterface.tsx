'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, FileText } from 'lucide-react'
import { ChatMessage } from './ChatMessage'
import { useMutation } from '@tanstack/react-query'
import axios from 'axios'

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
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || sendMessageMutation.isPending) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    sendMessageMutation.mutate(input)
    setInput('')
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <FileText className="w-16 h-16 text-text-secondary mx-auto mb-4 opacity-50" />
              <p className="text-text-secondary">
                Start a conversation by typing your question below
              </p>
            </div>
          )}

          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}

          {sendMessageMutation.isPending && (
            <div className="flex justify-start">
              <div className="flex items-center gap-2 px-4 py-3 bg-background-secondary rounded-lg">
                <Loader2 className="w-4 h-4 animate-spin text-brand-cyan" />
                <span className="text-sm text-text-secondary">Thinking...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-white/10 bg-background-secondary/50 backdrop-blur-lg p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="relative flex items-end gap-2">
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
              className="flex-1 min-h-[52px] max-h-32 px-4 py-3 bg-background-secondary border border-white/10 rounded-lg text-text-primary placeholder:text-text-secondary focus:outline-none focus:border-brand-cyan focus:ring-2 focus:ring-brand-cyan/20 transition-all duration-200 resize-none"
              rows={1}
            />
            <button
              type="submit"
              disabled={!input.trim() || sendMessageMutation.isPending}
              className="p-3 bg-gradient-brand rounded-lg hover:shadow-lg hover:shadow-brand-blue/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none"
            >
              {sendMessageMutation.isPending ? (
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
