'use client'

import { useState } from 'react'
import {
  BarChart3,
  TrendingUp,
  Activity,
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react'

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('7d')

  // Mock data - replace with real API calls
  const stats = {
    totalQueries: 1247,
    avgConfidence: 0.82,
    highConfidence: 856,
    mediumConfidence: 312,
    lowConfidence: 79,
    avgResponseTime: 2.3,
    domains: {
      healthcare: 423,
      legal: 312,
      finance: 298,
      general: 214
    }
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Analytics Dashboard</h1>
          <p className="text-text-secondary mt-1">
            AI performance metrics and insights
          </p>
        </div>

        <select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          className="input w-48"
        >
          <option value="24h">Last 24 Hours</option>
          <option value="7d">Last 7 Days</option>
          <option value="30d">Last 30 Days</option>
          <option value="90d">Last 90 Days</option>
        </select>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm">Total Queries</p>
              <p className="text-3xl font-bold mt-1">{stats.totalQueries.toLocaleString()}</p>
              <p className="text-green-500 text-sm mt-2 flex items-center gap-1">
                <TrendingUp className="w-4 h-4" />
                <span>+12.5% from last period</span>
              </p>
            </div>
            <div className="w-12 h-12 bg-brand-blue/20 rounded-lg flex items-center justify-center">
              <BarChart3 className="w-6 h-6 text-brand-cyan" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm">Avg Confidence</p>
              <p className="text-3xl font-bold mt-1">{(stats.avgConfidence * 100).toFixed(0)}%</p>
              <p className="text-green-500 text-sm mt-2 flex items-center gap-1">
                <TrendingUp className="w-4 h-4" />
                <span>+3.2% improvement</span>
              </p>
            </div>
            <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center">
              <CheckCircle className="w-6 h-6 text-green-500" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm">Avg Response Time</p>
              <p className="text-3xl font-bold mt-1">{stats.avgResponseTime}s</p>
              <p className="text-green-500 text-sm mt-2 flex items-center gap-1">
                <TrendingUp className="w-4 h-4" />
                <span>-15% faster</span>
              </p>
            </div>
            <div className="w-12 h-12 bg-brand-orange/20 rounded-lg flex items-center justify-center">
              <Clock className="w-6 h-6 text-brand-orange" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm">Low Confidence Alerts</p>
              <p className="text-3xl font-bold mt-1">{stats.lowConfidence}</p>
              <p className="text-yellow-500 text-sm mt-2 flex items-center gap-1">
                <AlertCircle className="w-4 h-4" />
                <span>Requires review</span>
              </p>
            </div>
            <div className="w-12 h-12 bg-yellow-500/20 rounded-lg flex items-center justify-center">
              <AlertCircle className="w-6 h-6 text-yellow-500" />
            </div>
          </div>
        </div>
      </div>

      {/* Confidence Distribution */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-brand-cyan" />
          Confidence Distribution
        </h2>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between mb-2">
              <span className="text-sm text-text-secondary">High Confidence (80%+)</span>
              <span className="text-sm font-medium">{stats.highConfidence} queries</span>
            </div>
            <div className="h-3 bg-background-tertiary rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full"
                style={{ width: `${(stats.highConfidence / stats.totalQueries) * 100}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <span className="text-sm text-text-secondary">Medium Confidence (50-80%)</span>
              <span className="text-sm font-medium">{stats.mediumConfidence} queries</span>
            </div>
            <div className="h-3 bg-background-tertiary rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-yellow-500 to-yellow-400 rounded-full"
                style={{ width: `${(stats.mediumConfidence / stats.totalQueries) * 100}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <span className="text-sm text-text-secondary">Low Confidence (&lt;50%)</span>
              <span className="text-sm font-medium">{stats.lowConfidence} queries</span>
            </div>
            <div className="h-3 bg-background-tertiary rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-red-500 to-red-400 rounded-full"
                style={{ width: `${(stats.lowConfidence / stats.totalQueries) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Domain Usage */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Domain Usage</h2>
          <div className="space-y-3">
            {Object.entries(stats.domains).map(([domain, count]) => (
              <div key={domain} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-brand-cyan" />
                  <span className="capitalize">{domain}</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-32 h-2 bg-background-tertiary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-brand rounded-full"
                      style={{ width: `${(count / stats.totalQueries) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium w-16 text-right">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Recent Trends</h2>
          <div className="space-y-4">
            <div className="flex items-start gap-3 p-3 bg-background-tertiary rounded-lg">
              <TrendingUp className="w-5 h-5 text-green-500 mt-0.5" />
              <div>
                <p className="font-medium">Healthcare queries increasing</p>
                <p className="text-sm text-text-secondary">
                  34% increase in medical research queries this week
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 bg-background-tertiary rounded-lg">
              <CheckCircle className="w-5 h-5 text-brand-cyan mt-0.5" />
              <div>
                <p className="font-medium">Improved response accuracy</p>
                <p className="text-sm text-text-secondary">
                  Average confidence score up 3.2% after model updates
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 bg-background-tertiary rounded-lg">
              <Activity className="w-5 h-5 text-brand-orange mt-0.5" />
              <div>
                <p className="font-medium">Peak usage hours</p>
                <p className="text-sm text-text-secondary">
                  Highest activity between 9 AM - 11 AM and 2 PM - 4 PM
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
