'use client'

import { ChatInterface } from '@/components/chat/ChatInterface'

export default function HomePage() {
  return (
    <div className="h-full flex flex-col">
      <ChatInterface />
    </div>
  )
}
