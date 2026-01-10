'use client'

import { Brain, FileSearch, Shield, Zap, TrendingUp, Globe } from 'lucide-react'

interface WelcomeScreenProps {
  onStartChat: () => void
}

const features = [
  {
    icon: Brain,
    title: 'AI-Powered Intelligence',
    description: 'Advanced RAG architecture with multi-model support'
  },
  {
    icon: FileSearch,
    title: 'Document Analysis',
    description: 'Upload and analyze documents with context-aware insights'
  },
  {
    icon: Shield,
    title: 'Privacy-First',
    description: 'Your data stays secure with explainable AI'
  },
  {
    icon: Zap,
    title: 'Real-Time Processing',
    description: 'Instant responses with streaming capabilities'
  },
  {
    icon: TrendingUp,
    title: 'Decision Intelligence',
    description: 'Data-driven insights for better decisions'
  },
  {
    icon: Globe,
    title: 'Global Knowledge',
    description: 'Access to worldwide information and research'
  }
]

const examplePrompts = [
  'Analyze this medical research paper for key findings',
  'Compare financial risks across these investment options',
  'Help me understand this legal contract',
  'Summarize the latest industry trends',
]

export function WelcomeScreen({ onStartChat }: WelcomeScreenProps) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8">
      <div className="max-w-5xl w-full space-y-12">
        {/* Hero Section */}
        <div className="text-center space-y-4">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="w-16 h-16 bg-gradient-brand rounded-2xl flex items-center justify-center animate-float">
              <Brain className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-5xl font-bold gradient-text">
            Welcome to Samādhān
          </h1>
          <p className="text-xl text-text-secondary max-w-2xl mx-auto">
            Your AI-powered decision intelligence platform. Ask questions, analyze documents,
            and get insights across healthcare, legal, finance, and more.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="card hover:scale-105 transition-transform duration-300"
            >
              <feature.icon className="w-8 h-8 text-brand-cyan mb-3" />
              <h3 className="font-semibold text-lg mb-2">{feature.title}</h3>
              <p className="text-sm text-text-secondary">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Example Prompts */}
        <div className="space-y-4">
          <h2 className="text-center text-lg font-semibold text-text-secondary">
            Try asking...
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {examplePrompts.map((prompt) => (
              <button
                key={prompt}
                onClick={onStartChat}
                className="text-left p-4 bg-background-secondary border border-white/10 rounded-lg hover:border-brand-cyan/50 hover:bg-background-tertiary transition-all duration-200 group"
              >
                <p className="text-sm text-text-secondary group-hover:text-text-primary transition-colors">
                  {prompt}
                </p>
              </button>
            ))}
          </div>
        </div>

        {/* CTA */}
        <div className="text-center">
          <button
            onClick={onStartChat}
            className="btn-primary text-lg px-8 py-3"
          >
            Start New Conversation
          </button>
        </div>
      </div>
    </div>
  )
}
