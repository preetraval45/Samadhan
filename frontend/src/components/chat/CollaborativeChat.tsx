'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Users, UserPlus, Circle, Send } from 'lucide-react'
import { EnhancedMessage } from './EnhancedMessage'

interface User {
  user_id: string
  username: string
  color?: string
  isOnline?: boolean
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  user_id?: string
  username?: string
  sources?: any[]
}

interface CollaborativeChatProps {
  conversationId: string
  currentUser: User
  onClose?: () => void
}

export function CollaborativeChat({
  conversationId,
  currentUser,
  onClose,
}: CollaborativeChatProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [activeUsers, setActiveUsers] = useState<User[]>([])
  const [typingUsers, setTypingUsers] = useState<Set<string>>(new Set())
  const [cursorPositions, setCursorPositions] = useState<Map<string, any>>(new Map())
  const [isConnected, setIsConnected] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Connect to WebSocket
  useEffect(() => {
    const wsUrl = `ws://localhost:8000/api/v1/collaborative/ws/collaborative/${conversationId}?user_id=${currentUser.user_id}&username=${currentUser.username}`

    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log('Connected to collaborative chat')
      setIsConnected(true)
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      switch (data.type) {
        case 'active_users':
          setActiveUsers(data.users)
          break

        case 'user_joined':
          console.log(`${data.username} joined`)
          setActiveUsers((prev) => [
            ...prev.filter((u) => u.user_id !== data.user_id),
            { user_id: data.user_id, username: data.username, isOnline: true },
          ])
          break

        case 'user_left':
          console.log(`${data.username} left`)
          setActiveUsers((prev) => prev.filter((u) => u.user_id !== data.user_id))
          break

        case 'message':
          const newMessage: Message = {
            id: data.message_id,
            role: data.role,
            content: data.content,
            timestamp: new Date(data.timestamp),
            user_id: data.user_id,
            username: data.username,
            sources: data.sources,
          }
          setMessages((prev) => [...prev, newMessage])
          break

        case 'typing':
          setTypingUsers((prev) => {
            const newSet = new Set(prev)
            if (data.is_typing && data.user_id !== currentUser.user_id) {
              newSet.add(data.username)
            } else {
              newSet.delete(data.username)
            }
            return newSet
          })
          break

        case 'cursor_position':
          if (data.user_id !== currentUser.user_id) {
            setCursorPositions((prev) => {
              const newMap = new Map(prev)
              newMap.set(data.user_id, {
                username: data.username,
                position: data.position,
              })
              return newMap
            })
          }
          break

        case 'reaction':
          // Handle message reactions
          console.log('Reaction:', data)
          break

        case 'highlight':
          // Handle text highlights
          console.log('Highlight:', data)
          break
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setIsConnected(false)
    }

    ws.onclose = () => {
      console.log('Disconnected from collaborative chat')
      setIsConnected(false)
    }

    wsRef.current = ws

    return () => {
      ws.close()
    }
  }, [conversationId, currentUser])

  const sendMessage = useCallback(
    (content: string) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: 'message',
            content,
          })
        )
      }
    },
    []
  )

  const sendTypingIndicator = useCallback(
    (isTyping: boolean) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: 'typing',
            is_typing: isTyping,
          })
        )
      }
    },
    []
  )

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)

    // Send typing indicator
    sendTypingIndicator(true)

    // Clear existing timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current)
    }

    // Set new timeout to stop typing indicator
    typingTimeoutRef.current = setTimeout(() => {
      sendTypingIndicator(false)
    }, 1000)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    // Add message locally
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
      user_id: currentUser.user_id,
      username: currentUser.username,
    }
    setMessages((prev) => [...prev, userMessage])

    // Send to WebSocket
    sendMessage(input)

    setInput('')
    sendTypingIndicator(false)
  }

  const getUserColor = (userId: string) => {
    const colors = [
      '#3B82F6', // blue
      '#10B981', // green
      '#F59E0B', // amber
      '#EF4444', // red
      '#8B5CF6', // purple
      '#EC4899', // pink
    ]
    const hash = userId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
    return colors[hash % colors.length]
  }

  return (
    <div className="flex h-full">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="px-4 py-3 bg-white dark:bg-background-secondary border-b border-gray-200 dark:border-white/10 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Users className="w-5 h-5 text-brand-cyan" />
              <h2 className="font-medium text-gray-900 dark:text-text-primary">
                Collaborative Chat
              </h2>
            </div>
            <div
              className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-xs ${
                isConnected
                  ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400'
                  : 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400'
              }`}
            >
              <Circle
                className={`w-2 h-2 fill-current ${
                  isConnected ? 'animate-pulse' : ''
                }`}
              />
              <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600 dark:text-text-secondary">
              {activeUsers.length} online
            </span>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto custom-scrollbar px-4 py-6 bg-gray-50 dark:bg-background">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map((message) => (
              <div key={message.id} className="relative">
                {/* Show username for collaborative messages */}
                {message.role === 'user' && message.username && (
                  <div className="text-xs font-medium mb-1 flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor: getUserColor(message.user_id || ''),
                      }}
                    />
                    <span
                      style={{ color: getUserColor(message.user_id || '') }}
                    >
                      {message.username}
                    </span>
                  </div>
                )}
                <EnhancedMessage message={message} />
              </div>
            ))}

            {/* Typing Indicators */}
            {typingUsers.size > 0 && (
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-text-secondary">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-brand-cyan rounded-full animate-bounce" />
                  <div
                    className="w-2 h-2 bg-brand-cyan rounded-full animate-bounce"
                    style={{ animationDelay: '0.1s' }}
                  />
                  <div
                    className="w-2 h-2 bg-brand-cyan rounded-full animate-bounce"
                    style={{ animationDelay: '0.2s' }}
                  />
                </div>
                <span>
                  {Array.from(typingUsers).join(', ')} {typingUsers.size === 1 ? 'is' : 'are'} typing...
                </span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 dark:border-white/10 bg-white dark:bg-background-secondary/50 backdrop-blur-lg p-4">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
            <div className="relative flex items-end gap-2">
              <textarea
                value={input}
                onChange={handleInputChange}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSubmit(e)
                  }
                }}
                placeholder="Type your message... (Shift + Enter for new line)"
                className="flex-1 min-h-[52px] max-h-32 px-4 py-3 bg-gray-50 dark:bg-background-secondary border border-gray-200 dark:border-white/10 rounded-lg text-gray-900 dark:text-text-primary placeholder:text-gray-500 dark:placeholder:text-text-secondary focus:outline-none focus:border-brand-cyan focus:ring-2 focus:ring-brand-cyan/20 transition-all duration-200 resize-none"
                rows={1}
                disabled={!isConnected}
              />
              <button
                type="submit"
                disabled={!input.trim() || !isConnected}
                className="p-3 bg-gradient-brand rounded-lg hover:shadow-lg hover:shadow-brand-blue/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none"
              >
                <Send className="w-5 h-5 text-white" />
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Sidebar - Active Users */}
      <div className="w-64 border-l border-gray-200 dark:border-white/10 bg-white dark:bg-background-secondary">
        <div className="p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium text-gray-900 dark:text-text-primary">
              Participants
            </h3>
            <button
              className="p-1.5 hover:bg-gray-100 dark:hover:bg-background rounded"
              title="Invite users"
            >
              <UserPlus className="w-4 h-4 text-gray-600 dark:text-text-secondary" />
            </button>
          </div>

          <div className="space-y-2">
            {activeUsers.map((user) => (
              <div
                key={user.user_id}
                className="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-background"
              >
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium"
                  style={{ backgroundColor: getUserColor(user.user_id) }}
                >
                  {user.username.charAt(0).toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-gray-900 dark:text-text-primary truncate">
                    {user.username}
                    {user.user_id === currentUser.user_id && (
                      <span className="ml-1 text-xs text-gray-500">(You)</span>
                    )}
                  </div>
                  <div className="flex items-center gap-1 text-xs text-green-600 dark:text-green-400">
                    <Circle className="w-2 h-2 fill-current" />
                    <span>Online</span>
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
