'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  MessageSquare,
  FileText,
  Upload,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Wand2,
  Settings,
  Bell,
  User,
  History
} from 'lucide-react'
import { clsx } from 'clsx'

const navigation = [
  { name: 'Chat', href: '/', icon: MessageSquare },
  { name: 'AI Studio', href: '/studio', icon: Wand2 },
  { name: 'History', href: '/history', icon: History },
  { name: 'Documents', href: '/documents', icon: FileText },
]

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const [showProfileMenu, setShowProfileMenu] = useState(false)
  const pathname = usePathname()

  return (
    <aside
      className={clsx(
        'relative flex flex-col bg-white dark:bg-background-secondary border-r border-gray-200 dark:border-white/10 transition-all duration-300',
        collapsed ? 'w-20' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-gray-200 dark:border-white/10">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-brand rounded-lg flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-lg bg-gradient-to-r from-brand-cyan to-brand-blue bg-clip-text text-transparent">Samādhān</span>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-2 hover:bg-gray-100 dark:hover:bg-background-tertiary rounded-lg transition-colors text-gray-600 dark:text-text-secondary"
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
                'hover:bg-gray-100 dark:hover:bg-background-tertiary group',
                isActive && 'bg-gradient-brand text-white shadow-lg shadow-brand-blue/30',
                !isActive && 'text-gray-600 dark:text-text-secondary hover:text-gray-900 dark:hover:text-text-primary'
              )}
              title={collapsed ? item.name : undefined}
            >
              <Icon
                className={clsx(
                  'w-5 h-5 transition-colors',
                  isActive && 'text-white',
                  !isActive && 'text-gray-600 dark:text-text-secondary group-hover:text-brand-cyan'
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
        <div className="p-4 border-t border-gray-200 dark:border-white/10">
          <button className="w-full bg-gradient-brand text-white px-4 py-2 rounded-lg hover:shadow-lg hover:shadow-brand-blue/50 transition-all duration-300 flex items-center justify-center gap-2 font-medium">
            <Upload className="w-4 h-4" />
            <span>Upload Document</span>
          </button>
        </div>
      )}

      {/* User Profile with Dropdown */}
      <div className="p-4 border-t border-gray-200 dark:border-white/10 relative">
        <button
          onClick={() => setShowProfileMenu(!showProfileMenu)}
          className={clsx(
            'w-full flex items-center gap-3 hover:bg-gray-100 dark:hover:bg-background-tertiary rounded-lg p-2 transition-colors',
            collapsed ? 'justify-center' : ''
          )}
        >
          <div className="w-10 h-10 bg-gradient-brand rounded-full flex items-center justify-center text-white font-bold">
            U
          </div>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 dark:text-text-primary truncate">
                User Name
              </p>
              <p className="text-xs text-gray-600 dark:text-text-secondary truncate">
                user@example.com
              </p>
            </div>
          )}
        </button>

        {/* Dropdown Menu */}
        {showProfileMenu && !collapsed && (
          <div className="absolute bottom-full left-4 right-4 mb-2 bg-white dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg shadow-lg overflow-hidden">
            <Link
              href="/profile"
              className="flex items-center gap-3 px-4 py-3 hover:bg-gray-100 dark:hover:bg-background-tertiary transition-colors text-gray-900 dark:text-text-primary"
            >
              <User className="w-4 h-4" />
              <span className="text-sm">Profile</span>
            </Link>
            <Link
              href="/notifications"
              className="flex items-center gap-3 px-4 py-3 hover:bg-gray-100 dark:hover:bg-background-tertiary transition-colors text-gray-900 dark:text-text-primary"
            >
              <Bell className="w-4 h-4" />
              <span className="text-sm">Notifications</span>
            </Link>
            <Link
              href="/settings"
              className="flex items-center gap-3 px-4 py-3 hover:bg-gray-100 dark:hover:bg-background-tertiary transition-colors text-gray-900 dark:text-text-primary"
            >
              <Settings className="w-4 h-4" />
              <span className="text-sm">Settings</span>
            </Link>
          </div>
        )}
      </div>
    </aside>
  )
}
