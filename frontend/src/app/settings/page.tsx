'use client'

import { useState } from 'react'
import {
  User,
  Bell,
  Shield,
  Palette,
  Database,
  Key,
  Save,
  Moon,
  Sun
} from 'lucide-react'

export default function SettingsPage() {
  const [theme, setTheme] = useState('dark')

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold gradient-text">Settings</h1>
        <p className="text-text-secondary mt-1">
          Manage your account and application preferences
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Settings Navigation */}
        <div className="lg:col-span-1">
          <div className="card space-y-2">
            {[
              { icon: User, label: 'Profile', active: true },
              { icon: Bell, label: 'Notifications', active: false },
              { icon: Palette, label: 'Appearance', active: false },
              { icon: Shield, label: 'Security', active: false },
              { icon: Key, label: 'API Keys', active: false },
              { icon: Database, label: 'Data & Privacy', active: false },
            ].map((item) => (
              <button
                key={item.label}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                  item.active
                    ? 'bg-gradient-brand text-white'
                    : 'hover:bg-background-tertiary'
                }`}
              >
                <item.icon className="w-5 h-5" />
                <span>{item.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Settings Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Profile Settings */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <User className="w-5 h-5 text-brand-cyan" />
              Profile Information
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Full Name</label>
                <input
                  type="text"
                  defaultValue="John Doe"
                  className="input w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Email</label>
                <input
                  type="email"
                  defaultValue="john.doe@example.com"
                  className="input w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Organization</label>
                <input
                  type="text"
                  defaultValue="Acme Corp"
                  className="input w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Role</label>
                <select className="input w-full">
                  <option>Data Analyst</option>
                  <option>Data Scientist</option>
                  <option>Engineer</option>
                  <option>Manager</option>
                  <option>Administrator</option>
                </select>
              </div>
            </div>
          </div>

          {/* Appearance */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Palette className="w-5 h-5 text-brand-cyan" />
              Appearance
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-3">Theme</label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setTheme('dark')}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      theme === 'dark'
                        ? 'border-brand-cyan bg-brand-cyan/10'
                        : 'border-white/10 hover:border-white/20'
                    }`}
                  >
                    <Moon className="w-6 h-6 mx-auto mb-2" />
                    <span className="text-sm">Dark</span>
                  </button>
                  <button
                    onClick={() => setTheme('light')}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      theme === 'light'
                        ? 'border-brand-cyan bg-brand-cyan/10'
                        : 'border-white/10 hover:border-white/20'
                    }`}
                  >
                    <Sun className="w-6 h-6 mx-auto mb-2" />
                    <span className="text-sm">Light (Coming Soon)</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Notifications */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Bell className="w-5 h-5 text-brand-cyan" />
              Notification Preferences
            </h2>

            <div className="space-y-4">
              {[
                { label: 'Email notifications', desc: 'Receive email updates about your activity' },
                { label: 'Query completion', desc: 'Notify when long-running queries finish' },
                { label: 'Weekly reports', desc: 'Get weekly summary of your usage' },
                { label: 'System updates', desc: 'Important platform updates and announcements' },
              ].map((item) => (
                <label key={item.label} className="flex items-start gap-3 cursor-pointer group">
                  <input
                    type="checkbox"
                    defaultChecked
                    className="mt-1 w-5 h-5 rounded border-white/20 bg-background-tertiary text-brand-cyan focus:ring-brand-cyan focus:ring-offset-0"
                  />
                  <div className="flex-1">
                    <p className="font-medium group-hover:text-brand-cyan transition-colors">
                      {item.label}
                    </p>
                    <p className="text-sm text-text-secondary">{item.desc}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Save Button */}
          <div className="flex justify-end">
            <button className="btn-primary">
              <Save className="w-4 h-4 mr-2" />
              Save Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
