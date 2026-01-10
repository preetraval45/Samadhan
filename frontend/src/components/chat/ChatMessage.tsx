'use client'

import { Bot, User, ExternalLink } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

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

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-gradient-brand flex items-center justify-center flex-shrink-0">
          <Bot className="w-5 h-5 text-white" />
        </div>
      )}

      <div className={`flex flex-col gap-2 max-w-[80%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`px-4 py-3 rounded-2xl ${
            isUser
              ? 'bg-gradient-brand text-white'
              : 'bg-background-secondary border border-white/10'
          }`}
        >
          {isUser ? (
            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Sources */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="w-full space-y-2">
            <p className="text-xs text-text-secondary font-medium">Sources:</p>
            <div className="grid gap-2">
              {message.sources.map((source, index) => (
                <div
                  key={index}
                  className="glass rounded-lg p-3 text-xs space-y-1 hover:border-brand-cyan/30 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <ExternalLink className="w-3 h-3 text-brand-cyan" />
                      <span className="font-medium text-text-primary">
                        {source.document}
                      </span>
                    </div>
                    <span className="text-text-secondary">
                      Page {source.page} • {Math.round(source.relevance_score * 100)}% match
                    </span>
                  </div>
                  <p className="text-text-secondary leading-relaxed">
                    {source.snippet}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        <span className="text-xs text-text-secondary">
          {message.timestamp.toLocaleTimeString()}
        </span>
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-background-tertiary flex items-center justify-center flex-shrink-0">
          <User className="w-5 h-5 text-brand-cyan" />
        </div>
      )}
    </div>
  )
}
