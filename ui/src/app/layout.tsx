'use client'; // Marked as client to ensure consistent behavior, though next.js handles strict mode
import './globals.css';
import Sidebar from '@/components/Sidebar';
import { ThemeProvider } from '@/components/ThemeProvider';
import ConfirmModal from '@/components/ConfirmModal';
import { Suspense } from 'react';
import AuthWrapper from '@/components/AuthWrapper';
import DocModal from '@/components/DocModal';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // Authentication is strictly disabled
  const authRequired = false;

  return (
    <html lang="en" className="dark">
      <head>
        <meta name="apple-mobile-web-app-title" content="AI-Toolkit" />
      </head>
      <body className={inter.className}>
        <ThemeProvider>
          <AuthWrapper authRequired={authRequired}>
            <div className="flex h-screen bg-gray-950">
              <Sidebar />
              <main className="flex-1 overflow-auto bg-gray-950 text-gray-100 relative">
                <Suspense>{children}</Suspense>
              </main>
            </div>
          </AuthWrapper>
        </ThemeProvider>
        <ConfirmModal />
        <DocModal />
      </body>
    </html>
  );
}