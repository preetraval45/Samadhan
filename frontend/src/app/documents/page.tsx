'use client'

import { useState } from 'react'
import {
  Upload,
  FileText,
  Trash2,
  Download,
  Search,
  Filter,
  CheckCircle,
  Clock,
  AlertCircle
} from 'lucide-react'

interface Document {
  id: string
  filename: string
  size: number
  type: string
  status: 'processing' | 'completed' | 'failed'
  uploadedAt: Date
  processedAt?: Date
  chunks: number
  domain?: string
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([
    {
      id: '1',
      filename: 'Medical_Research_2024.pdf',
      size: 2457600,
      type: 'application/pdf',
      status: 'completed',
      uploadedAt: new Date('2024-01-09'),
      processedAt: new Date('2024-01-09'),
      chunks: 45,
      domain: 'healthcare'
    },
    {
      id: '2',
      filename: 'Contract_NDA_Template.docx',
      size: 856000,
      type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      status: 'completed',
      uploadedAt: new Date('2024-01-08'),
      processedAt: new Date('2024-01-08'),
      chunks: 12,
      domain: 'legal'
    },
    {
      id: '3',
      filename: 'Financial_Report_Q4.xlsx',
      size: 1245000,
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      status: 'processing',
      uploadedAt: new Date('2024-01-10'),
      chunks: 0,
      domain: 'finance'
    }
  ])

  const [searchTerm, setSearchTerm] = useState('')
  const [selectedDomain, setSelectedDomain] = useState('all')

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  const getStatusIcon = (status: Document['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'processing':
        return <Clock className="w-5 h-5 text-yellow-500 animate-spin" />
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-500" />
    }
  }

  const getStatusText = (status: Document['status']) => {
    switch (status) {
      case 'completed':
        return 'Processed'
      case 'processing':
        return 'Processing...'
      case 'failed':
        return 'Failed'
    }
  }

  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = doc.filename.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesDomain = selectedDomain === 'all' || doc.domain === selectedDomain
    return matchesSearch && matchesDomain
  })

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold gradient-text">Document Management</h1>
        <p className="text-text-secondary mt-1">
          Upload and manage documents for AI-powered analysis
        </p>
      </div>

      {/* Upload Area */}
      <div className="card border-2 border-dashed border-white/20 hover:border-brand-cyan/50 transition-colors">
        <div className="flex flex-col items-center justify-center py-12">
          <div className="w-16 h-16 bg-gradient-brand rounded-2xl flex items-center justify-center mb-4">
            <Upload className="w-8 h-8 text-white" />
          </div>
          <h3 className="text-lg font-semibold mb-2">Upload Documents</h3>
          <p className="text-text-secondary text-sm mb-4 text-center max-w-md">
            Drag and drop files here or click to browse
            <br />
            Supported formats: PDF, DOCX, TXT, CSV, JSON (Max 10MB)
          </p>
          <button className="btn-primary">
            <Upload className="w-4 h-4 mr-2" />
            Choose Files
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <p className="text-text-secondary text-sm">Total Documents</p>
          <p className="text-2xl font-bold mt-1">{documents.length}</p>
        </div>
        <div className="card">
          <p className="text-text-secondary text-sm">Processed</p>
          <p className="text-2xl font-bold mt-1 text-green-500">
            {documents.filter(d => d.status === 'completed').length}
          </p>
        </div>
        <div className="card">
          <p className="text-text-secondary text-sm">Processing</p>
          <p className="text-2xl font-bold mt-1 text-yellow-500">
            {documents.filter(d => d.status === 'processing').length}
          </p>
        </div>
        <div className="card">
          <p className="text-text-secondary text-sm">Total Chunks</p>
          <p className="text-2xl font-bold mt-1">
            {documents.reduce((sum, d) => sum + d.chunks, 0)}
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary" />
          <input
            type="text"
            placeholder="Search documents..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input pl-10 w-full"
          />
        </div>
        <select
          value={selectedDomain}
          onChange={(e) => setSelectedDomain(e.target.value)}
          className="input w-full sm:w-48"
        >
          <option value="all">All Domains</option>
          <option value="healthcare">Healthcare</option>
          <option value="legal">Legal</option>
          <option value="finance">Finance</option>
          <option value="general">General</option>
        </select>
      </div>

      {/* Documents List */}
      <div className="card">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">
                  Document
                </th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">
                  Domain
                </th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">
                  Size
                </th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">
                  Status
                </th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">
                  Chunks
                </th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">
                  Uploaded
                </th>
                <th className="text-right py-3 px-4 text-sm font-medium text-text-secondary">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredDocuments.map((doc) => (
                <tr
                  key={doc.id}
                  className="border-b border-white/10 hover:bg-background-tertiary transition-colors"
                >
                  <td className="py-4 px-4">
                    <div className="flex items-center gap-3">
                      <FileText className="w-5 h-5 text-brand-cyan" />
                      <span className="font-medium">{doc.filename}</span>
                    </div>
                  </td>
                  <td className="py-4 px-4">
                    <span className="px-2 py-1 bg-brand-blue/20 text-brand-cyan text-xs rounded-full capitalize">
                      {doc.domain || 'general'}
                    </span>
                  </td>
                  <td className="py-4 px-4 text-sm text-text-secondary">
                    {formatFileSize(doc.size)}
                  </td>
                  <td className="py-4 px-4">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(doc.status)}
                      <span className="text-sm">{getStatusText(doc.status)}</span>
                    </div>
                  </td>
                  <td className="py-4 px-4 text-sm">
                    {doc.chunks > 0 ? doc.chunks : '-'}
                  </td>
                  <td className="py-4 px-4 text-sm text-text-secondary">
                    {doc.uploadedAt.toLocaleDateString()}
                  </td>
                  <td className="py-4 px-4">
                    <div className="flex items-center justify-end gap-2">
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
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {filteredDocuments.length === 0 && (
            <div className="text-center py-12">
              <FileText className="w-16 h-16 text-text-secondary mx-auto mb-4 opacity-50" />
              <p className="text-text-secondary">No documents found</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
