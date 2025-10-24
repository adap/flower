import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import Link from 'next/link';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'My Chat App',
  description: 'Chat using API, Server-side, and Client-side approaches',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gray-50 dark:bg-gray-900`}
      >
        <header className="bg-white dark:bg-gray-800 shadow-sm">
          <nav className="container mx-auto flex justify-between items-center py-4 px-6">
            <Link href="/" className="text-gray-700 text-xl font-bold hover:text-gray-500">
              Home
            </Link>
            <div className="flex items-center space-x-8">
              <Link
                href="/api-chat"
                className="text-gray-700 dark:text-gray-300 hover:text-gray-600"
              >
                API Chat
              </Link>
              <Link
                href="/server-side-chat"
                className="text-gray-700 dark:text-gray-300 hover:text-gray-600"
              >
                Server-side Chat
              </Link>
              <Link
                href="/client-side-chat"
                className="text-gray-700 dark:text-gray-300 hover:text-gray-600"
              >
                Client-side Chat
              </Link>
              <Link
                href="/client-side-no-history-chat"
                className="text-gray-700 dark:text-gray-300 hover:text-gray-600"
              >
                Client-side No History Chat
              </Link>
            </div>
          </nav>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
