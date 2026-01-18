'use client'

import { useState } from 'react'
import { User, Bot, ExternalLink } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import mermaid from 'mermaid'
import { useEffect, useRef } from 'react'
import 'katex/dist/katex.min.css'

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
  model?: string
  attachments?: Array<{
    name: string
    type: string
    size: number
    preview?: string
  }>
}

interface EnhancedMessageProps {
  message: Message
}

// Initialize mermaid
mermaid.initialize({
  startOnLoad: true,
  theme: 'dark',
  securityLevel: 'loose',
})

export function EnhancedMessage({ message }: EnhancedMessageProps) {
  const [expandedSources, setExpandedSources] = useState(false)
  const mermaidRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (mermaidRef.current) {
      mermaid.contentLoaded()
    }
  }, [message.content])

  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} group`}>
      <div
        className={`flex gap-3 max-w-[85%] ${
          isUser ? 'flex-row-reverse' : 'flex-row'
        }`}
      >
        {/* Avatar */}
        <div
          className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
            isUser
              ? 'bg-gradient-brand'
              : 'bg-gradient-to-br from-purple-500 to-pink-500'
          }`}
        >
          {isUser ? (
            <User className="w-4 h-4 text-white" />
          ) : (
            <Bot className="w-4 h-4 text-white" />
          )}
        </div>

        {/* Message Content */}
        <div className="flex-1 min-w-0">
          <div
            className={`px-4 py-3 rounded-2xl ${
              isUser
                ? 'bg-gradient-brand text-white'
                : 'bg-white dark:bg-background-secondary border border-gray-200 dark:border-white/10'
            }`}
          >
            {/* Model Badge */}
            {!isUser && message.model && (
              <div className="mb-2">
                <span className="inline-block px-2 py-0.5 bg-brand-cyan/10 text-brand-cyan text-xs rounded-full">
                  {message.model} model
                </span>
              </div>
            )}

            {/* Message Text with Enhanced Rendering */}
            <div
              className={`prose prose-sm max-w-none ${
                isUser
                  ? 'prose-invert'
                  : 'dark:prose-invert prose-headings:text-gray-900 dark:prose-headings:text-text-primary prose-p:text-gray-700 dark:prose-p:text-text-secondary'
              }`}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex]}
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '')
                    const language = match ? match[1] : ''

                    // Handle Mermaid diagrams
                    if (language === 'mermaid') {
                      return (
                        <div
                          ref={mermaidRef}
                          className="mermaid bg-white dark:bg-background p-4 rounded-lg my-4"
                        >
                          {String(children).replace(/\n$/, '')}
                        </div>
                      )
                    }

                    // Handle code blocks
                    return !inline && match ? (
                      <div className="relative group/code">
                        <div className="absolute right-2 top-2 opacity-0 group-hover/code:opacity-100 transition-opacity">
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(String(children))
                            }}
                            className="px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded"
                          >
                            Copy
                          </button>
                        </div>
                        <SyntaxHighlighter
                          style={vscDarkPlus}
                          language={language}
                          PreTag="div"
                          {...props}
                          customStyle={{
                            margin: '1rem 0',
                            borderRadius: '0.5rem',
                          }}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      </div>
                    ) : (
                      <code
                        className={`${className} px-1.5 py-0.5 bg-gray-100 dark:bg-background rounded text-sm ${
                          isUser ? 'bg-white/20' : ''
                        }`}
                        {...props}
                      >
                        {children}
                      </code>
                    )
                  },
                  // Enhanced table styling
                  table({ children, ...props }) {
                    return (
                      <div className="overflow-x-auto my-4">
                        <table
                          className="min-w-full divide-y divide-gray-200 dark:divide-white/10"
                          {...props}
                        >
                          {children}
                        </table>
                      </div>
                    )
                  },
                  // Enhanced link styling
                  a({ href, children, ...props }) {
                    return (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-brand-cyan hover:text-brand-blue underline inline-flex items-center gap-1"
                        {...props}
                      >
                        {children}
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>

            {/* Edited Badge */}
            {message.isEdited && (
              <div className="mt-2 text-xs text-gray-500 dark:text-text-secondary">
                (edited)
              </div>
            )}

            {/* Attachments */}
            {message.attachments && message.attachments.length > 0 && (
              <div className="mt-3 space-y-2">
                {message.attachments.map((att, index) => (
                  <div
                    key={index}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg ${
                      isUser
                        ? 'bg-white/10'
                        : 'bg-gray-50 dark:bg-background border border-gray-200 dark:border-white/10'
                    }`}
                  >
                    <div className="flex-1">
                      <div
                        className={`text-sm font-medium ${
                          isUser ? 'text-white' : 'text-gray-900 dark:text-text-primary'
                        }`}
                      >
                        {att.name}
                      </div>
                      <div className="text-xs text-gray-400">
                        {(att.size / 1024).toFixed(1)} KB
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Sources */}
            {!isUser && message.sources && message.sources.length > 0 && (
              <div className="mt-3 pt-3 border-t border-gray-200 dark:border-white/10">
                <button
                  onClick={() => setExpandedSources(!expandedSources)}
                  className="text-sm text-brand-cyan hover:text-brand-blue font-medium"
                >
                  {expandedSources ? 'Hide' : 'Show'} {message.sources.length}{' '}
                  sources
                </button>

                {expandedSources && (
                  <div className="mt-2 space-y-2">
                    {message.sources.map((source, index) => (
                      <div
                        key={index}
                        className="p-3 bg-gray-50 dark:bg-background rounded-lg border border-gray-200 dark:border-white/10"
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium text-gray-900 dark:text-text-primary">
                            {source.document}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-text-secondary">
                            Page {source.page} •{' '}
                            {(source.relevance_score * 100).toFixed(0)}% relevant
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-text-secondary">
                          {source.snippet}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Timestamp */}
          <div
            className={`mt-1 text-xs text-gray-500 dark:text-text-secondary ${
              isUser ? 'text-right' : 'text-left'
            }`}
          >
            {message.timestamp.toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
