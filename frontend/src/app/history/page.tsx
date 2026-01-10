'use client'

import { useState } from 'react'
import { Clock, Search, Trash2, Download, Filter } from 'lucide-react'

interface Conversation {
  id: string
  title: string
  timestamp: Date
  messageCount: number
  domain: string
  preview: string
}

export default function HistoryPage() {
  const [conversations] = useState<Conversation[]>([
    {
      id: '1',
      title: 'Medical Research Analysis',
      timestamp: new Date('2024-01-10T10:30:00'),
      messageCount: 12,
      domain: 'healthcare',
      preview: 'Analyzed clinical trial data for cardiovascular treatment...'
    },
    {
      id: '2',
      title: 'Contract Review - NDA',
      timestamp: new Date('2024-01-09T15:45:00'),
      messageCount: 8,
      domain: 'legal',
      preview: 'Reviewed non-disclosure agreement for compliance issues...'
    },
    {
      id: '3',
      title: 'Portfolio Risk Assessment',
      timestamp: new Date('2024-01-09T09:15:00'),
      messageCount: 15,
      domain: 'finance',
      preview: 'Evaluated investment portfolio risk metrics and recommendations...'
    },
    {
      id: '4',
      title: 'Drug Interaction Check',
      timestamp: new Date('2024-01-08T14:20:00'),
      messageCount: 6,
      domain: 'healthcare',
      preview: 'Checked potential drug interactions for patient medications...'
    },
    {
      id: '5',
      title: 'Quarterly Financial Report',
      timestamp: new Date('2024-01-07T11:00:00'),
      messageCount: 20,
      domain: 'finance',
      preview: 'Generated comprehensive financial analysis for Q4...'
    }
  ])

  const getDomainColor = (domain: string) => {
    const colors = {
      healthcare: 'bg-green-500/20 text-green-500',
      legal: 'bg-blue-500/20 text-blue-500',
      finance: 'bg-orange-500/20 text-orange-500',
      general: 'bg-gray-500/20 text-gray-500'
    }
    return colors[domain as keyof typeof colors] || colors.general
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold gradient-text">Conversation History</h1>
        <p className="text-text-secondary mt-1">
          View and manage your past conversations
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <p className="text-text-secondary text-sm">Total Conversations</p>
          <p className="text-2xl font-bold mt-1">{conversations.length}</p>
        </div>
        <div className="card">
          <p className="text-text-secondary text-sm">This Week</p>
          <p className="text-2xl font-bold mt-1">12</p>
        </div>
        <div className="card">
          <p className="text-text-secondary text-sm">Avg Messages</p>
          <p className="text-2xl font-bold mt-1">
            {Math.round(conversations.reduce((sum, c) => sum + c.messageCount, 0) / conversations.length)}
          </p>
        </div>
        <div className="card">
          <p className="text-text-secondary text-sm">Most Used</p>
          <p className="text-2xl font-bold mt-1">Healthcare</p>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary" />
          <input
            type="text"
            placeholder="Search conversations..."
            className="input pl-10 w-full"
          />
        </div>
        <button className="btn-secondary">
          <Filter className="w-4 h-4 mr-2" />
          Filters
        </button>
      </div>

      {/* Conversations List */}
      <div className="space-y-3">
        {conversations.map((conversation) => (
          <div
            key={conversation.id}
            className="card hover:border-brand-cyan/50 transition-all cursor-pointer group"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="font-semibold text-lg group-hover:text-brand-cyan transition-colors">
                    {conversation.title}
                  </h3>
                  <span className={`px-2 py-1 rounded-full text-xs capitalize ${getDomainColor(conversation.domain)}`}>
                    {conversation.domain}
                  </span>
                </div>

                <p className="text-sm text-text-secondary mb-3">
                  {conversation.preview}
                </p>

                <div className="flex items-center gap-4 text-sm text-text-secondary">
                  <div className="flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    <span>{conversation.timestamp.toLocaleDateString()}</span>
                  </div>
                  <span>{conversation.messageCount} messages</span>
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  className="p-2 hover:bg-background-tertiary rounded-lg transition-colors"
                  title="Download"
                >
                  <Download className="w-4 h-4 text-text-secondary hover:text-brand-cyan" />
                </button>
                <button
                  className="p-2 hover:bg-background-tertiary rounded-lg transition-colors"
                  title="Delete"
                >
                  <Trash2 className="w-4 h-4 text-text-secondary hover:text-red-500" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Load More */}
      <div className="text-center">
        <button className="btn-secondary">
          Load More Conversations
        </button>
      </div>
    </div>
  )
}
