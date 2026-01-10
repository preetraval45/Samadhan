'use client'

import { useState } from 'react'
import {
  Brain,
  Zap,
  DollarSign,
  Clock,
  CheckCircle,
  Settings,
  Play,
  Pause
} from 'lucide-react'

interface Model {
  id: string
  name: string
  provider: string
  status: 'active' | 'inactive'
  capabilities: string[]
  contextLength: number
  costPer1kTokens: number
  avgLatency: number
  accuracy: number
}

export default function ModelsPage() {
  const [models] = useState<Model[]>([
    {
      id: 'gpt-4',
      name: 'GPT-4',
      provider: 'OpenAI',
      status: 'active',
      capabilities: ['text', 'code', 'analysis'],
      contextLength: 8192,
      costPer1kTokens: 0.03,
      avgLatency: 2.1,
      accuracy: 94.5
    },
    {
      id: 'gpt-4-turbo',
      name: 'GPT-4 Turbo',
      provider: 'OpenAI',
      status: 'active',
      capabilities: ['text', 'code', 'vision', 'analysis'],
      contextLength: 128000,
      costPer1kTokens: 0.01,
      avgLatency: 1.8,
      accuracy: 95.2
    },
    {
      id: 'claude-3-opus',
      name: 'Claude 3 Opus',
      provider: 'Anthropic',
      status: 'active',
      capabilities: ['text', 'code', 'vision', 'analysis'],
      contextLength: 200000,
      costPer1kTokens: 0.015,
      avgLatency: 2.5,
      accuracy: 96.1
    },
    {
      id: 'claude-3-sonnet',
      name: 'Claude 3 Sonnet',
      provider: 'Anthropic',
      status: 'active',
      capabilities: ['text', 'code', 'vision'],
      contextLength: 200000,
      costPer1kTokens: 0.003,
      avgLatency: 1.4,
      accuracy: 93.8
    }
  ])

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold gradient-text">AI Models</h1>
        <p className="text-text-secondary mt-1">
          Manage and monitor LLM models
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm">Active Models</p>
              <p className="text-2xl font-bold mt-1">{models.filter(m => m.status === 'active').length}</p>
            </div>
            <CheckCircle className="w-10 h-10 text-green-500 opacity-20" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm">Avg Latency</p>
              <p className="text-2xl font-bold mt-1">
                {(models.reduce((sum, m) => sum + m.avgLatency, 0) / models.length).toFixed(1)}s
              </p>
            </div>
            <Clock className="w-10 h-10 text-brand-cyan opacity-20" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm">Avg Accuracy</p>
              <p className="text-2xl font-bold mt-1">
                {(models.reduce((sum, m) => sum + m.accuracy, 0) / models.length).toFixed(1)}%
              </p>
            </div>
            <Zap className="w-10 h-10 text-brand-orange opacity-20" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm">Est. Monthly Cost</p>
              <p className="text-2xl font-bold mt-1">$247</p>
            </div>
            <DollarSign className="w-10 h-10 text-green-500 opacity-20" />
          </div>
        </div>
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {models.map((model) => (
          <div key={model.id} className="card hover:border-brand-cyan/50 transition-all duration-300">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-gradient-brand rounded-xl flex items-center justify-center">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-bold text-lg">{model.name}</h3>
                  <p className="text-sm text-text-secondary">{model.provider}</p>
                </div>
              </div>
              <button
                className={`p-2 rounded-lg transition-colors ${
                  model.status === 'active'
                    ? 'bg-green-500/20 text-green-500 hover:bg-green-500/30'
                    : 'bg-red-500/20 text-red-500 hover:bg-red-500/30'
                }`}
              >
                {model.status === 'active' ? (
                  <Play className="w-5 h-5" />
                ) : (
                  <Pause className="w-5 h-5" />
                )}
              </button>
            </div>

            {/* Capabilities */}
            <div className="flex flex-wrap gap-2 mb-4">
              {model.capabilities.map((cap) => (
                <span
                  key={cap}
                  className="px-2 py-1 bg-brand-blue/20 text-brand-cyan text-xs rounded-full capitalize"
                >
                  {cap}
                </span>
              ))}
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-xs text-text-secondary mb-1">Context Length</p>
                <p className="font-semibold">{model.contextLength.toLocaleString()} tokens</p>
              </div>
              <div>
                <p className="text-xs text-text-secondary mb-1">Cost / 1K Tokens</p>
                <p className="font-semibold">${model.costPer1kTokens}</p>
              </div>
              <div>
                <p className="text-xs text-text-secondary mb-1">Avg Latency</p>
                <p className="font-semibold">{model.avgLatency}s</p>
              </div>
              <div>
                <p className="text-xs text-text-secondary mb-1">Accuracy</p>
                <p className="font-semibold">{model.accuracy}%</p>
              </div>
            </div>

            {/* Accuracy Bar */}
            <div className="mb-4">
              <div className="flex justify-between text-xs text-text-secondary mb-1">
                <span>Accuracy Score</span>
                <span>{model.accuracy}%</span>
              </div>
              <div className="h-2 bg-background-tertiary rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-brand rounded-full transition-all duration-500"
                  style={{ width: `${model.accuracy}%` }}
                />
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-2">
              <button className="flex-1 btn-secondary text-sm py-2">
                <Settings className="w-4 h-4 mr-2" />
                Configure
              </button>
              <button className="flex-1 btn-primary text-sm py-2">
                <Zap className="w-4 h-4 mr-2" />
                Test Model
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Performance Comparison */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Performance Comparison</h2>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span>Latency</span>
              <span className="text-text-secondary">Lower is better</span>
            </div>
            {models.map((model) => (
              <div key={model.id} className="flex items-center gap-3 mb-2">
                <span className="w-32 text-sm truncate">{model.name}</span>
                <div className="flex-1 h-6 bg-background-tertiary rounded-lg overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-lg flex items-center px-2"
                    style={{ width: `${(model.avgLatency / 3) * 100}%` }}
                  >
                    <span className="text-xs text-white font-medium">{model.avgLatency}s</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div>
            <div className="flex justify-between text-sm mb-2">
              <span>Accuracy</span>
              <span className="text-text-secondary">Higher is better</span>
            </div>
            {models.map((model) => (
              <div key={model.id} className="flex items-center gap-3 mb-2">
                <span className="w-32 text-sm truncate">{model.name}</span>
                <div className="flex-1 h-6 bg-background-tertiary rounded-lg overflow-hidden">
                  <div
                    className="h-full bg-gradient-brand rounded-lg flex items-center px-2"
                    style={{ width: `${model.accuracy}%` }}
                  >
                    <span className="text-xs text-white font-medium">{model.accuracy}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
