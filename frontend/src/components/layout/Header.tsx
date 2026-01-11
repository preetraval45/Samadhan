'use client'

import { Search, Bell, HelpCircle } from 'lucide-react'
import { ThemeToggle } from '../theme/ThemeToggle'

export function Header() {
  return (
    <header className="h-16 border-b border-gray-200 dark:border-white/10 bg-white dark:bg-background-secondary/50 backdrop-blur-lg">
      <div className="h-full px-6 flex items-center justify-between">
        {/* Search */}
        <div className="flex-1 max-w-2xl">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 dark:text-text-secondary" />
            <input
              type="text"
              placeholder="Search conversations, documents, or ask a question..."
              className="w-full pl-10 pr-4 py-2 bg-gray-50 dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg text-gray-900 dark:text-text-primary placeholder:text-gray-500 dark:placeholder:text-text-secondary focus:outline-none focus:border-brand-cyan focus:ring-2 focus:ring-brand-cyan/20 transition-all duration-200"
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 ml-4">
          <ThemeToggle />

          <button
            className="p-2 hover:bg-gray-100 dark:hover:bg-background-tertiary rounded-lg transition-colors relative"
            aria-label="Notifications"
          >
            <Bell className="w-5 h-5 text-gray-600 dark:text-text-secondary hover:text-gray-900 dark:hover:text-text-primary transition-colors" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-brand-orange rounded-full"></span>
          </button>

          <button
            className="p-2 hover:bg-gray-100 dark:hover:bg-background-tertiary rounded-lg transition-colors"
            aria-label="Help"
          >
            <HelpCircle className="w-5 h-5 text-gray-600 dark:text-text-secondary hover:text-gray-900 dark:hover:text-text-primary transition-colors" />
          </button>
        </div>
      </div>
    </header>
  )
}
