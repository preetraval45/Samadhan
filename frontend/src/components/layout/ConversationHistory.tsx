'use client'

import { MessageSquare, Plus, Trash2, Edit2 } from 'lucide-react'
import { useState, useEffect } from 'react'

interface Conversation {
  id: string
  title: string
  timestamp: Date
  preview: string
}

export function ConversationHistory() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversation, setActiveConversation] = useState<string | null>(null)

  useEffect(() => {
    // Load conversations from localStorage
    const saved = localStorage.getItem('conversations')
    if (saved) {
      const parsed = JSON.parse(saved)
      setConversations(parsed.map((c: any) => ({
        ...c,
        timestamp: new Date(c.timestamp)
      })))
    }
  }, [])

  const formatDate = (date: Date) => {
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))

    if (days === 0) return 'Today'
    if (days === 1) return 'Yesterday'
    if (days < 7) return `${days} days ago`
    return date.toLocaleDateString()
  }

  const deleteConversation = (id: string) => {
    const updated = conversations.filter(c => c.id !== id)
    setConversations(updated)
    localStorage.setItem('conversations', JSON.stringify(updated))
  }

  return (
    <div className="w-64 bg-white dark:bg-background-secondary border-r border-gray-200 dark:border-white/10 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-white/10">
        <button
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-brand-cyan text-white rounded-lg hover:bg-brand-cyan/90 transition-colors"
          onClick={() => {
            // Create new conversation
            setActiveConversation(null)
          }}
        >
          <Plus className="w-4 h-4" />
          <span className="font-medium">New Chat</span>
        </button>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {conversations.length === 0 ? (
          <div className="p-4 text-center text-gray-500 dark:text-text-secondary">
            <MessageSquare className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No conversations yet</p>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {conversations.map((conversation) => (
              <div
                key={conversation.id}
                className={`group relative p-3 rounded-lg cursor-pointer transition-all ${
                  activeConversation === conversation.id
                    ? 'bg-brand-cyan/10 border border-brand-cyan/30'
                    : 'hover:bg-gray-100 dark:hover:bg-background-tertiary'
                }`}
                onClick={() => setActiveConversation(conversation.id)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-medium text-gray-900 dark:text-text-primary truncate">
                      {conversation.title}
                    </h3>
                    <p className="text-xs text-gray-500 dark:text-text-secondary truncate mt-1">
                      {conversation.preview}
                    </p>
                    <p className="text-xs text-gray-400 dark:text-text-tertiary mt-1">
                      {formatDate(conversation.timestamp)}
                    </p>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      className="p-1 hover:bg-gray-200 dark:hover:bg-background rounded"
                      onClick={(e) => {
                        e.stopPropagation()
                        // Edit conversation title
                      }}
                      aria-label="Edit conversation"
                    >
                      <Edit2 className="w-3 h-3 text-gray-600 dark:text-text-secondary" />
                    </button>
                    <button
                      className="p-1 hover:bg-red-100 dark:hover:bg-red-900/20 rounded"
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteConversation(conversation.id)
                      }}
                      aria-label="Delete conversation"
                    >
                      <Trash2 className="w-3 h-3 text-red-600 dark:text-red-400" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
