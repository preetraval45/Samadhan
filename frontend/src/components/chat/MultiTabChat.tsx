'use client'

import { useState, useCallback } from 'react'
import { Plus, X, Pin, Folder, Maximize2 } from 'lucide-react'
import { EnhancedChatInterface } from './EnhancedChatInterface'
import { DragDropContext, Droppable, Draggable, DropResult } from '@hello-pangea/dnd'

interface ChatTab {
  id: string
  title: string
  isPinned: boolean
  groupId?: string
  lastMessage?: string
  createdAt: Date
}

interface TabGroup {
  id: string
  name: string
  color: string
}

export function MultiTabChat() {
  const [tabs, setTabs] = useState<ChatTab[]>([
    {
      id: '1',
      title: 'New Chat',
      isPinned: false,
      createdAt: new Date(),
    },
  ])
  const [activeTabId, setActiveTabId] = useState('1')
  const [groups, setGroups] = useState<TabGroup[]>([])
  const [isSplitScreen, setIsSplitScreen] = useState(false)
  const [splitTabId, setSplitTabId] = useState<string | null>(null)

  const createNewTab = useCallback(() => {
    const newTab: ChatTab = {
      id: Date.now().toString(),
      title: `New Chat ${tabs.length + 1}`,
      isPinned: false,
      createdAt: new Date(),
    }
    setTabs((prev) => [...prev, newTab])
    setActiveTabId(newTab.id)
  }, [tabs.length])

  const closeTab = useCallback((tabId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setTabs((prev) => {
      const filtered = prev.filter((t) => t.id !== tabId)
      if (filtered.length === 0) {
        // Always keep at least one tab
        return [{
          id: Date.now().toString(),
          title: 'New Chat',
          isPinned: false,
          createdAt: new Date(),
        }]
      }
      return filtered
    })

    if (activeTabId === tabId) {
      const currentIndex = tabs.findIndex((t) => t.id === tabId)
      const nextTab = tabs[currentIndex + 1] || tabs[currentIndex - 1]
      if (nextTab) {
        setActiveTabId(nextTab.id)
      }
    }
  }, [activeTabId, tabs])

  const togglePin = useCallback((tabId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setTabs((prev) =>
      prev.map((t) =>
        t.id === tabId ? { ...t, isPinned: !t.isPinned } : t
      )
    )
  }, [])

  const handleDragEnd = (result: DropResult) => {
    if (!result.destination) return

    const items = Array.from(tabs)
    const [reorderedItem] = items.splice(result.source.index, 1)
    items.splice(result.destination.index, 0, reorderedItem)

    setTabs(items)
  }

  const toggleSplitScreen = useCallback(() => {
    if (!isSplitScreen) {
      // Find another tab to show in split screen
      const otherTab = tabs.find((t) => t.id !== activeTabId)
      if (otherTab) {
        setSplitTabId(otherTab.id)
      }
    } else {
      setSplitTabId(null)
    }
    setIsSplitScreen(!isSplitScreen)
  }, [isSplitScreen, activeTabId, tabs])

  const updateTabTitle = (tabId: string, title: string) => {
    setTabs((prev) =>
      prev.map((t) =>
        t.id === tabId ? { ...t, title } : t
      )
    )
  }

  const activeTab = tabs.find((t) => t.id === activeTabId)
  const splitTab = splitTabId ? tabs.find((t) => t.id === splitTabId) : null

  return (
    <div className="flex flex-col h-full">
      {/* Tab Bar */}
      <div className="flex items-center bg-white dark:bg-background border-b border-gray-200 dark:border-white/10">
        <DragDropContext onDragEnd={handleDragEnd}>
          <Droppable droppableId="tabs" direction="horizontal">
            {(provided) => (
              <div
                {...provided.droppableProps}
                ref={provided.innerRef}
                className="flex-1 flex items-center overflow-x-auto custom-scrollbar"
              >
                {tabs.map((tab, index) => (
                  <Draggable key={tab.id} draggableId={tab.id} index={index}>
                    {(provided, snapshot) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        {...provided.dragHandleProps}
                        className={`
                          group relative flex items-center gap-2 px-4 py-3 border-r border-gray-200 dark:border-white/10
                          cursor-pointer transition-all min-w-[180px] max-w-[240px]
                          ${
                            activeTabId === tab.id
                              ? 'bg-gray-50 dark:bg-background-secondary border-b-2 border-b-brand-cyan'
                              : 'hover:bg-gray-50 dark:hover:bg-background-secondary/50'
                          }
                          ${snapshot.isDragging ? 'shadow-lg opacity-75' : ''}
                        `}
                        onClick={() => setActiveTabId(tab.id)}
                      >
                        {tab.isPinned && (
                          <Pin className="w-3 h-3 text-brand-cyan flex-shrink-0" />
                        )}

                        <span className="flex-1 text-sm text-gray-900 dark:text-text-primary truncate">
                          {tab.title}
                        </span>

                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={(e) => togglePin(tab.id, e)}
                            className="p-1 hover:bg-gray-200 dark:hover:bg-background rounded"
                            title={tab.isPinned ? 'Unpin' : 'Pin'}
                          >
                            <Pin className={`w-3 h-3 ${tab.isPinned ? 'text-brand-cyan' : 'text-gray-500'}`} />
                          </button>

                          {tabs.length > 1 && (
                            <button
                              onClick={(e) => closeTab(tab.id, e)}
                              className="p-1 hover:bg-red-100 dark:hover:bg-red-900/20 rounded"
                              title="Close tab"
                            >
                              <X className="w-3 h-3 text-gray-500 dark:text-text-secondary hover:text-red-500" />
                            </button>
                          )}
                        </div>
                      </div>
                    )}
                  </Draggable>
                ))}
                {provided.placeholder}
              </div>
            )}
          </Droppable>
        </DragDropContext>

        {/* Tab Actions */}
        <div className="flex items-center gap-2 px-3 border-l border-gray-200 dark:border-white/10">
          <button
            onClick={createNewTab}
            className="p-2 hover:bg-gray-100 dark:hover:bg-background-secondary rounded transition-colors"
            title="New tab (Ctrl+T)"
          >
            <Plus className="w-4 h-4 text-gray-600 dark:text-text-secondary" />
          </button>

          <button
            onClick={toggleSplitScreen}
            className={`p-2 rounded transition-colors ${
              isSplitScreen
                ? 'bg-brand-cyan/10 text-brand-cyan'
                : 'hover:bg-gray-100 dark:hover:bg-background-secondary text-gray-600 dark:text-text-secondary'
            }`}
            title="Split screen"
          >
            <Maximize2 className="w-4 h-4" />
          </button>

          <button
            className="p-2 hover:bg-gray-100 dark:hover:bg-background-secondary rounded transition-colors"
            title="Tab groups"
          >
            <Folder className="w-4 h-4 text-gray-600 dark:text-text-secondary" />
          </button>
        </div>
      </div>

      {/* Chat Content */}
      <div className="flex-1 flex overflow-hidden">
        {isSplitScreen && splitTab ? (
          <>
            {/* Left Chat */}
            <div className="flex-1 border-r border-gray-200 dark:border-white/10 overflow-hidden">
              <div className="h-full flex flex-col">
                <div className="px-4 py-2 bg-gray-50 dark:bg-background-secondary border-b border-gray-200 dark:border-white/10">
                  <h3 className="text-sm font-medium text-gray-900 dark:text-text-primary">
                    {activeTab?.title}
                  </h3>
                </div>
                <EnhancedChatInterface
                  tabId={activeTabId}
                  onTitleUpdate={(title) => updateTabTitle(activeTabId, title)}
                />
              </div>
            </div>

            {/* Right Chat */}
            <div className="flex-1 overflow-hidden">
              <div className="h-full flex flex-col">
                <div className="px-4 py-2 bg-gray-50 dark:bg-background-secondary border-b border-gray-200 dark:border-white/10">
                  <h3 className="text-sm font-medium text-gray-900 dark:text-text-primary">
                    {splitTab.title}
                  </h3>
                </div>
                <EnhancedChatInterface
                  tabId={splitTabId}
                  onTitleUpdate={(title) => updateTabTitle(splitTabId, title)}
                />
              </div>
            </div>
          </>
        ) : (
          <EnhancedChatInterface
            tabId={activeTabId}
            onTitleUpdate={(title) => updateTabTitle(activeTabId, title)}
          />
        )}
      </div>
    </div>
  )
}
