import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from '@/components/providers'
import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Samādhān - Decision Intelligence Platform',
  description: 'Multi-Modal AI-Powered Research & Decision Intelligence Platform',
  keywords: ['AI', 'Decision Intelligence', 'RAG', 'Machine Learning', 'Enterprise AI'],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${inter.className} bg-white dark:bg-background transition-colors duration-200`}>
        <Providers>
          <div className="flex h-screen overflow-hidden bg-white dark:bg-background">
            {/* Sidebar */}
            <Sidebar />

            {/* Main Content */}
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Header */}
              <Header />

              {/* Page Content */}
              <main className="flex-1 overflow-y-auto custom-scrollbar bg-gray-50 dark:bg-background">
                {children}
              </main>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  )
}
