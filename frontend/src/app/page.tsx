'use client'

import { useState } from 'react'
import { ChatInterface } from '@/components/chat/ChatInterface'
import { WelcomeScreen } from '@/components/chat/WelcomeScreen'

export default function HomePage() {
  const [hasStartedChat, setHasStartedChat] = useState(false)

  return (
    <div className="h-full flex flex-col">
      {!hasStartedChat ? (
        <WelcomeScreen onStartChat={() => setHasStartedChat(true)} />
      ) : (
        <ChatInterface />
      )}
    </div>
  )
}
