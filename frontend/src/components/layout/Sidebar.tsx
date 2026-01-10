'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  MessageSquare,
  FileText,
  Database,
  Settings,
  BarChart3,
  Brain,
  Upload,
  History,
  ChevronLeft,
  ChevronRight,
  Sparkles
} from 'lucide-react'
import { clsx } from 'clsx'

const navigation = [
  { name: 'Chat', href: '/', icon: MessageSquare },
  { name: 'Documents', href: '/documents', icon: FileText },
  { name: 'Knowledge Base', href: '/knowledge', icon: Database },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Models', href: '/models', icon: Brain },
  { name: 'History', href: '/history', icon: History },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const pathname = usePathname()

  return (
    <aside
      className={clsx(
        'relative flex flex-col bg-background-secondary border-r border-white/10 transition-all duration-300',
        collapsed ? 'w-20' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-white/10">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-brand rounded-lg flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-lg gradient-text">Samādhān</span>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-2 hover:bg-background-tertiary rounded-lg transition-colors"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? (
            <ChevronRight className="w-5 h-5" />
          ) : (
            <ChevronLeft className="w-5 h-5" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2 custom-scrollbar overflow-y-auto">
        {navigation.map((item) => {
          const isActive = pathname === item.href
          const Icon = item.icon

          return (
            <Link
              key={item.name}
              href={item.href}
              className={clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200',
                'hover:bg-background-tertiary group',
                isActive && 'bg-gradient-brand text-white shadow-lg shadow-brand-blue/30',
                !isActive && 'text-text-secondary hover:text-text-primary'
              )}
              title={collapsed ? item.name : undefined}
            >
              <Icon
                className={clsx(
                  'w-5 h-5 transition-colors',
                  isActive && 'text-white',
                  !isActive && 'text-text-secondary group-hover:text-brand-cyan'
                )}
              />
              {!collapsed && (
                <span className="font-medium">{item.name}</span>
              )}
            </Link>
          )
        })}
      </nav>

      {/* Quick Actions */}
      {!collapsed && (
        <div className="p-4 border-t border-white/10">
          <button className="w-full btn-primary flex items-center justify-center gap-2">
            <Upload className="w-4 h-4" />
            <span>Upload Document</span>
          </button>
        </div>
      )}

      {/* User Profile */}
      <div className="p-4 border-t border-white/10">
        <div className={clsx(
          'flex items-center gap-3',
          collapsed ? 'justify-center' : ''
        )}>
          <div className="w-10 h-10 bg-gradient-brand rounded-full flex items-center justify-center text-white font-bold">
            U
          </div>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-text-primary truncate">
                User Name
              </p>
              <p className="text-xs text-text-secondary truncate">
                user@example.com
              </p>
            </div>
          )}
        </div>
      </div>
    </aside>
  )
}
