'use client'

import { useState } from 'react'
import { Network, Search, Plus, TrendingUp, Link as LinkIcon } from 'lucide-react'

export default function KnowledgePage() {
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Knowledge Graph</h1>
          <p className="text-text-secondary mt-1">
            Explore entity relationships and organizational knowledge
          </p>
        </div>
        <button className="btn-primary">
          <Plus className="w-4 h-4 mr-2" />
          Add Entity
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm">Total Entities</p>
              <p className="text-2xl font-bold mt-1">1,247</p>
            </div>
            <Network className="w-10 h-10 text-brand-cyan opacity-20" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm">Relationships</p>
              <p className="text-2xl font-bold mt-1">3,842</p>
            </div>
            <LinkIcon className="w-10 h-10 text-brand-orange opacity-20" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm">Active Queries</p>
              <p className="text-2xl font-bold mt-1">156</p>
            </div>
            <TrendingUp className="w-10 h-10 text-green-500 opacity-20" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-text-secondary text-sm">Graph Depth</p>
              <p className="text-2xl font-bold mt-1">12 levels</p>
            </div>
            <Network className="w-10 h-10 text-brand-blue opacity-20" />
          </div>
        </div>
      </div>

      {/* Search */}
      <div className="card">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary" />
          <input
            type="text"
            placeholder="Search entities, relationships, or concepts..."
            className="input pl-10 w-full"
          />
        </div>
      </div>

      {/* Graph Visualization Placeholder */}
      <div className="card min-h-[500px] flex items-center justify-center">
        <div className="text-center">
          <Network className="w-24 h-24 text-brand-cyan mx-auto mb-4 opacity-30" />
          <h3 className="text-xl font-semibold mb-2">Knowledge Graph Visualization</h3>
          <p className="text-text-secondary max-w-md mx-auto">
            Interactive graph visualization will appear here. Connect Neo4j to start exploring
            entity relationships and patterns.
          </p>
          <button className="btn-primary mt-4">
            Initialize Graph
          </button>
        </div>
      </div>

      {/* Entity Types */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <h3 className="font-semibold mb-4">Recent Entities</h3>
          <div className="space-y-3">
            {['Dr. Smith', 'ABC Hospital', 'Cardiology Dept', 'Patient Care Protocol'].map((entity) => (
              <div key={entity} className="flex items-center gap-3 p-2 hover:bg-background-tertiary rounded-lg transition-colors cursor-pointer">
                <div className="w-2 h-2 rounded-full bg-brand-cyan" />
                <span className="text-sm">{entity}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h3 className="font-semibold mb-4">Top Relationships</h3>
          <div className="space-y-3">
            {['WORKS_FOR', 'MANAGES', 'COLLABORATES_WITH', 'SPECIALIZES_IN'].map((rel) => (
              <div key={rel} className="flex items-center justify-between p-2 hover:bg-background-tertiary rounded-lg transition-colors cursor-pointer">
                <span className="text-sm">{rel}</span>
                <span className="text-xs text-text-secondary">142</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h3 className="font-semibold mb-4">Graph Metrics</h3>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-text-secondary">Avg Connections</span>
              <span className="font-medium">3.2</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-text-secondary">Max Depth</span>
              <span className="font-medium">12</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-text-secondary">Clusters</span>
              <span className="font-medium">8</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-text-secondary">Orphan Nodes</span>
              <span className="font-medium">23</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
